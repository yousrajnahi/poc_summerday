import streamlit as st
from src.utils import initialize_vector_store
from src.document_processing import (create_directory, save_uploaded_files, organize_files_by_extension,
                                     load_documents, split_documents, remove_directory, convert_files_in_directory)
from langchain_community.document_loaders import (UnstructuredFileLoader, PDFMinerLoader, CSVLoader, JSONLoader,
                                                  TextLoader, UnstructuredXMLLoader, UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader, UnstructuredEmailLoader, WikipediaLoader)

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
st.set_page_config(page_title="Document Upload",page_icon="📊",layout="wide",initial_sidebar_state="expanded") 
st.title("Document Upload")

# Initialize vector store
db, index, namespace, embeddings = initialize_vector_store()
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
    'csv': {"csv_args": {"delimiter": ",", "quotechar": '"'}},
    'json': {'jq_schema': '.[] | "MainTask: \(.MainTask), MainTaskSummary: \(.MainTaskSummary), Tips: \(.Tips)  "'},
    'default': None
}
# Define a mapping from file extensions to loader classes
text_splitter_map = {
    'default': RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True),
    'ipynb': RecursiveCharacterTextSplitter.from_language(Language.PYTHON,chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True) ,
    'py': RecursiveCharacterTextSplitter.from_language(Language.PYTHON,chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True) ,
    'md': RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN,chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True) ,
    'mdx': RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN,chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True) ,
    'html': RecursiveCharacterTextSplitter.from_language(Language.HTML,chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True) ,
}
# Temporary directory for file processing
directory = "./Internal_data"
create_directory(directory)
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        save_uploaded_files(uploaded_files, directory)
        convert_files_in_directory(directory)
        extensions_dict = organize_files_by_extension(directory)
        for ext, files in extensions_dict.items():
            glob_pattern = f'**/*.{ext}'
            loader_class = loader_cls_map.get(ext, loader_cls_map['default'])
            loader_args = loader_kwargs_map.get(ext, loader_kwargs_map['default'])
            documents = load_documents(directory, glob_pattern, loader_class, loader_args)
            chunks = split_documents(documents,text_splitter_map,ext)
          
            db.add_documents(chunks)
            st.toast(str(ext) + ' docs added successfully', icon="✅")
    # Clean up
    remove_directory(directory)
    st.success("All documents processed and added to the vector store.")
  
# Ajout de la partie Wikipedia
wikipedia_query = st.text_input("Enter a Wikipedia query (optional):")
if wikipedia_query:
    with st.spinner("Loading Wikipedia content..."):
        loader = WikipediaLoader(query=wikipedia_query, load_max_docs=2, load_all_available_meta=False, lang='en')
        wikipedia_documents = loader.load()
        
        # Utiliser le même text splitter que pour les autres documents
        wikipedia_chunks = split_documents(wikipedia_documents, text_splitter_map, 'default')
        
        db.add_documents(wikipedia_chunks)
        st.success(f"Wikipedia content for '{wikipedia_query}' added to the vector store.")
  # New YouTube part
youtube_url = st.text_input("Enter a YouTube video URL (optional):")
if youtube_url:
    with st.spinner("Loading YouTube content..."):
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False, language='en')
        youtube_documents = loader.load()
        
        # Use the same text splitter as for other documents
        youtube_chunks = split_documents(youtube_documents, text_splitter_map, 'default')
        
        db.add_documents(youtube_chunks)
        st.success(f"YouTube content from '{youtube_url}' added to the vector store.")

# New arXiv part
arxiv_query = st.text_input("Enter an arXiv query (optional):")
if arxiv_query:
    with st.spinner("Loading arXiv content..."):
        loader = ArxivLoader(query=arxiv_query, load_all_available_meta=False, load_max_docs=2)
        arxiv_documents = loader.load()
        
        # Use the same text splitter as for other documents
        arxiv_chunks = split_documents(arxiv_documents, text_splitter_map, 'default')
        
        db.add_documents(arxiv_chunks)
        st.success(f"arXiv content for '{arxiv_query}' added to the vector store.")
