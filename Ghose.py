import streamlit as st
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from together import Together

# Title and Header
st.title("Ghose Ai")
st.header("Chat with your docs")
st.write("")

# Initialize Together API
api_key = "92e7c27a78ee8122ac243a1d8cba697e5b5d2fbe8afbbed938f4b8eaa32cda7e"  # Replace with your API key
client = Together(api_key=api_key)

# Paths to FAISS index and metadata
faiss_index_path = "DataBase/ayodhya_faiss.index"  # Path to FAISS index
metadata_path = "DataBase/ayodhya_metadata.json"  # Path to metadata JSON

@st.cache_resource
def load_faiss_and_metadata():
    try:
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Error loading FAISS index or metadata: {e}")
        return None, None

# Load FAISS and metadata
index, metadata = load_faiss_and_metadata()

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Load Sentence Transformer model
embedding_model = load_embedding_model()

# Retrieve similar chunks
def retrieve_similar_chunks(query, index, metadata, model, k=3):
    try:
        query_embedding = model.encode(query, convert_to_tensor=False)
        distances, indices = index.search(np.array([query_embedding]), k)
        results = [{"chunk": metadata[i]["chunk"], "distance": distances[0][idx]} for idx, i in enumerate(indices[0])]
        return results
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return []

# Input query from user
query = st.text_input("Enter your question:")

if query:
    if index is None or metadata is None or embedding_model is None:
        st.error("The app failed to initialize required resources. Please check the setup.")
    else:
        # Retrieve context
        with st.spinner("Retrieving context..."):
            retrieved_chunks = retrieve_similar_chunks(query, index, metadata, embedding_model, k=3)

        # # Display retrieved context
        # if retrieved_chunks:
        #     st.subheader("Retrieved Context")
        #     for i, chunk in enumerate(retrieved_chunks):
        #         st.markdown(f"**Chunk {i+1}:** {chunk['chunk']} (Distance: {chunk['distance']:.4f})")
        # else:
        #     st.error("No relevant context could be retrieved.")

        # Construct prompt for Together API
        retrieved_text = "\n".join([chunk["chunk"] for chunk in retrieved_chunks])
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in legal documents."},
            {"role": "user", "content": f"Context:\n{retrieved_text}\n\nQuestion:\n{query}\n\nAnswer:"}
        ]

        # Call LLaMA API for a response
        if retrieved_chunks:
            with st.spinner("Generating response..."):
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7,
                        top_p=0.7,
                        top_k=50,
                        repetition_penalty=1,
                        stop=["<|eot_id|>", "<|eom_id|>"],
                        stream=True
                    )

                    # Buffer to accumulate the complete response
                    llama_response = ""
                    for token in response:
                        if hasattr(token, 'choices'):
                            llama_response += token.choices[0].delta.content
                    
                    # Display only the final response
                    st.subheader("Response")
                    st.write(llama_response)

                except Exception as e:
                    st.error(f"Error during LLaMA API call: {e}")
