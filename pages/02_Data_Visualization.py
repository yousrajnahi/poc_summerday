import streamlit as st
from src.utils import initialize_vector_store
from src.vector_store import get_data_in_vector_store
from src.displays import vectordb_to_dfdb, df_visualisation, create_2d_embeddings


st.title("Data Visualization")

# Initialize vector store
db, index, namespace = initialize_vector_store()

if st.sidebar.button("View Data"):
    with st.spinner("Fetching and processing data..."):
        embeddings, chunks, vectors, vector_ids = get_data_in_vector_store(index, namespace)
        documents_projected = create_2d_embeddings(embeddings)
        df = vectordb_to_dfdb(documents_projected, chunks, vectors, vector_ids)
        
    st.write("Data from Vector Store:")
    st.write(df)
    
    with st.spinner("Generating visualization..."):
        fig = df_visualisation(df)
    
    st.plotly_chart(fig)

