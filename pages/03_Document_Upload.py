import streamlit as st
from src.vector_store import get_vector_store
from src.embeddings import get_embeddings
from src.document_processing import create_directory, save_uploaded_files, organize_files_by_extension, load_documents, split_documents, remove_directory
from langchain_community.document_loaders import (UnstructuredFileLoader, PDFMinerLoader, CSVLoader, JSONLoader,
                                                  TextLoader, UnstructuredXMLLoader, UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader, UnstructuredEmailLoader)
from pinecone import ServerlessSpec

st.title("Document Upload")

# Initialize vector store and embeddings
index_name = "docs-rag-summerday"
namespace = "summerday-space"
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = get_embeddings(embedding_model_name)
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)

# File uploader
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

# Loader mappings
loader_cls_map = {
    'md': UnstructuredMarkdownLoader,
    'txt': TextLoader,
    'pdf': PDFMinerLoader,
    'xml': UnstructuredXMLLoader,
    'html': UnstructuredHTMLLoader,
    'eml': UnstructuredEmailLoader,
    'csv': CSVLoader,
    'json': JSONLoader,
    'default': UnstructuredFileLoader
}

loader_kwargs_map = {
    'pdf': None,
    'txt': None,
    'csv': {"csv_args": {"delimiter": ",", "quotechar": '"'}},
    'json': {'jq_schema': '.[] | "MainTask: \(.MainTask), MainTaskSummary: \(.MainTaskSummary), Tips: \(.Tips)  "'},
    'default': {'strategy': "fast"}
}

# Temporary directory for file processing
directory = "./Internal_data"
create_directory(directory)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        save_uploaded_files(uploaded_files, directory)
        extensions_dict = organize_files_by_extension(directory)
        for ext, files in extensions_dict.items():
            glob_pattern = f'**/*.{ext}'
            loader_class = loader_cls_map.get(ext, loader_cls_map['default'])
            loader_args = loader_kwargs_map.get(ext, loader_kwargs_map['default'])
            documents = load_documents(directory, glob_pattern, loader_class, loader_args)
            chunks = split_documents(documents)
            db.add_documents(chunks)
            st.success(f"{ext.upper()} documents added successfully")

    # Clean up
    remove_directory(directory)
    st.success("All documents processed and added to the vector store.")
