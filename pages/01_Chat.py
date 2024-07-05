import streamlit as st
import os
from src.utils import initialize_vector_store, get_or_create_retrieval_chain
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

st.title("Chat")

# Configure your environment
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY'] 
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY'] 

# Sidebar Clear Chat Button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages.clear()
    st.rerun()

st.sidebar.markdown("### Prompt Parameters")

# Sidebar for model and chain type selection
chain_types = ['stuff', "refine", "map_reduce", "map_rerank"]
default_chain_type = 'refine'
selected_chain_type = st.sidebar.selectbox("Choose a chain type:", options=chain_types, index = chain_types.index(default_chain_type))

# Add a divider
st.sidebar.divider()

st.sidebar.markdown("### Model Parameters")

model_options = ["gemma-7b-it", "mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"]
default_model = "llama3-70b-8192"
selected_model = st.sidebar.selectbox("Choose a model:", options=model_options, index=model_options.index(default_model))

# Add temperature slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
# Add a divider
st.sidebar.divider()

# Add search type selection and parameter inputs in the sidebar
st.sidebar.markdown("### Retriever Parameters")

search_types = ['similarity', 'similarity_score_threshold', 'mmr']
default_search_type = 'mmr'
search_type = st.sidebar.selectbox("Choose a search type:", options = search_types, index= search_types.index(default_search_type))
default_k = 4
k_value = st.sidebar.slider("Number of documents to retrieve (k)", min_value=1, max_value=20, value= default_k)

# Define search_kwargs based on the selected search type
search_kwargs = {'k': k_value}
if search_type == 'similarity_score_threshold':
    score_threshold = st.sidebar.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    search_kwargs['score_threshold'] = score_threshold
elif search_type == 'mmr':
    lambda_mult = st.sidebar.slider("Diversity of results (lambda_mult)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    search_kwargs['lambda_mult'] = lambda_mult

# Initialize vector store
db, index, namespace, embeddings = initialize_vector_store()

# Get or create retrieval chain
retrieval_chain = get_or_create_retrieval_chain(selected_chain_type, selected_model, db, search_type, search_kwargs, temperature)


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
        st.session_state['last_query'] = prompt
    
    with st.chat_message("assistant"):
        if len(prompt.split()) > 500:
            st.warning(f"Your input is too long and will be truncated to fit the model's limit of 500 tokens.")
            prompt = ' '.join(prompt.split()[:500])
        
        response = retrieval_chain.run(prompt)
        query_vector = embeddings.embed_query(prompt)
        query_vector_2d = np.array([query_vector], dtype=np.float32)
        st.session_state['last_query_vector'] = query_vector
        
        matching_docs = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs).get_relevant_documents(prompt)
        st.session_state['matching_docs'] = matching_docs
        
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


