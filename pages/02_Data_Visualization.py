import streamlit as st
from src.vector_store import get_vector_store, get_data_in_vector_store
from src.embeddings import get_embeddings
from src.displays import vectordb_to_dfdb, df_visualisation, create_2d_embeddings
from pinecone import ServerlessSpec

st.title("Data Visualization")

# Initialize vector store and embeddings
index_name = "docs-rag-summerday"
namespace = "summerday-space"
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = get_embeddings(embedding_model_name)
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)

if st.button("View Data"):
    with st.spinner("Fetching and processing data..."):
        embeddings, chunks, vectors, vector_ids = get_data_in_vector_store(index, namespace)
        documents_projected = create_2d_embeddings(embeddings)
        df = vectordb_to_dfdb(documents_projected, chunks, vectors, vector_ids)
        
    st.write("Data from Vector Store:")
    st.write(df)
    
    with st.spinner("Generating visualization..."):
        fig = df_visualisation(df)
    
    st.plotly_chart(fig)
