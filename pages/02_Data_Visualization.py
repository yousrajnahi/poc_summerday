import streamlit as st
from src.utils import initialize_vector_store
from src.vector_store import get_data_in_vector_store
from src.displays import vectordb_to_dfdb, df_visualisation, create_2d_embeddings
import numpy as np
np.random.seed(42)

st.title("Data Visualization")

# Initialize vector store
db, index, namespace, embeddings = initialize_vector_store()

if st.sidebar.button("View Data") or 'data_viz_df' in st.session_state:
    if 'data_viz_df' not in st.session_state:
        with st.spinner("Fetching and processing data..."):
            embeddings, chunks, vectors, vector_ids = get_data_in_vector_store(index, namespace)
            
            if 'last_query' in st.session_state and 'last_query_vector' in st.session_state:
                embeddings = np.append(embeddings, [st.session_state['last_query_vector']], axis=0)
                chunks.append({"source": "User query", "text": st.session_state['last_query']})
                vectors.append(st.session_state['last_query_vector'])
                vector_ids.append("id_user_query")
                
            documents_projected = create_2d_embeddings(embeddings)
            
            matching_docs = st.session_state.get('matching_docs', None)
            df = vectordb_to_dfdb(documents_projected, chunks, vectors, vector_ids, matching_docs)
            
            if 'last_query' in st.session_state:
                df.loc[df['id'] == 'id_user_query', 'symbol'] = 'star'
                df.loc[df['id'] == 'id_user_query', 'size_col'] = 4
            
            st.session_state['data_viz_df'] = df
    else:
        df = st.session_state['data_viz_df']

    st.write("Data from Vector Store:")
    st.write(df)
    
    with st.spinner("Generating visualization..."):
        fig = df_visualisation(df)
    
    st.plotly_chart(fig)

if st.sidebar.button("Clear Visualization"):
    if 'data_viz_df' in st.session_state:
        del st.session_state['data_viz_df']
    st.rerun()

if 'last_query' in st.session_state:
    st.sidebar.write(f"Last query: {st.session_state['last_query']}")
