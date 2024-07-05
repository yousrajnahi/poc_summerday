import streamlit as st
import os
from src.utils import initialize_vector_store, get_or_create_retrieval_chain
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

st.title("Chat")

# Configure your environment
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY'] 
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY'] 

# Sidebar for model and chain type selection
chain_types = ['stuff', "refine", "map_reduce", "map_rerank"]
selected_chain_type = st.sidebar.selectbox("Choose a chain type:", options=chain_types)

model_options = ["gemma-7b-it", "mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"]
default_model = "llama3-70b-8192"
selected_model = st.sidebar.selectbox("Choose a model:", options=model_options, index=model_options.index(default_model))

# Initialize vector store
db, index, namespace, embeddings = initialize_vector_store()

# Get or create retrieval chain
retrieval_chain = get_or_create_retrieval_chain(selected_chain_type, selected_model, db)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if len(prompt.split()) > 500:
            st.warning(f"Your input is too long and will be truncated to fit the model's limit of 500 tokens.")
            prompt = ' '.join(prompt.split()[:500])
        
        response = retrieval_chain.run(prompt)
        query_vector = embeddings.embed_query(prompt)
        query_vector_2d = np.array([query_vector], dtype=np.float32)
        matching_docs = db.as_retriever(search_type='mmr',search_kwargs={'k': 6, 'lambda_mult': 0.25}).get_relevant_documents(prompt)
        matching_docs_vectors = np.array([embeddings.embed_documents([doc.page_content])[0] for doc in matching_docs])
        scores = list(cosine_similarity(query_vector_2d, matching_docs_vectors)[0])
        sources = [doc.metadata.get("source", doc.metadata) for doc in matching_docs]
        
        # Merge response with sources
        full_response = response + "\n\n### Sources:\n"
        for source, score in zip(sources, scores):
            similarity_percentage = score * 100
            full_response += f"- {source}: {similarity_percentage:.2f}%\n"
        
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar Clear Chat Button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages.clear()
    st.rerun()
