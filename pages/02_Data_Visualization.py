import streamlit as st
from src.utils import initialize_vector_store
from src.vector_store import get_data_in_vector_store
from src.displays import vectordb_to_dfdb, df_visualisation, create_2d_embeddings

st.title("Data Visualization")

# Initialize vector store
db, index, namespace, embeddings = initialize_vector_store()

if st.sidebar.button("View Data"):
    with st.spinner("Fetching and processing data..."):
        embeddings, chunks, vectors, vector_ids = get_data_in_vector_store(index, namespace)
        
        # Check if there's a last query to include
        if 'last_query' in st.session_state and 'last_query_vector' in st.session_state:
            embeddings.append(st.session_state['last_query_vector'])
            chunks.append({"source": "User query", "text": st.session_state['last_query']})
            vectors.append(st.session_state['last_query_vector'])
            vector_ids.append("user_query")

        documents_projected = create_2d_embeddings(embeddings)
        df = vectordb_to_dfdb(documents_projected, chunks, vectors, vector_ids)
        
        # Mark the user query point if it exists
        if 'last_query' in st.session_state:
            df.loc[df['id'] == 'user_query', 'symbol'] = 'star'
            df.loc[df['id'] == 'user_query', 'size_col'] = 10  # Make user query point larger
        
    st.write("Data from Vector Store:")
    st.write(df)
    
    with st.spinner("Generating visualization..."):
        fig = df_visualisation(df)
    
    st.plotly_chart(fig)

# Optionally, display the last query
if 'last_query' in st.session_state:
    st.sidebar.write(f"Last query: {st.session_state['last_query']}")
