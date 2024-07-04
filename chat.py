import streamlit as st
from src.utils import initialize_vector_store

# Initialize vector store once
initialize_vector_store()

st.title("RAG - AI 4 CI")
st.write("Welcome to our application. Use the sidebar to navigate between pages.")

st.markdown("""
## Features:
- Chat with AI using different models and chain types
- Visualize data from our vector store
- Upload and process various document types

Please select a page from the sidebar to get started.
""")
