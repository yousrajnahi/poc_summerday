__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.regex import RegexParser
from langchain_community.document_loaders import (UnstructuredFileLoader,
                                                  PDFMinerLoader,
                                                  CSVLoader,
                                                  JSONLoader,
                                                  TextLoader,
                                                  UnstructuredXMLLoader,
                                                  UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredEmailLoader)

import warnings
import os
import shutil
from src.vector_store import get_vector_store, get_data_in_vector_store
from src.embeddings import get_embeddings
from src.displays import create_2d_embeddings, vectordb_to_dfdb, df_visualisation
from src.document_processing import organize_files_by_extension,  load_documents, split_documents, save_uploaded_files, create_directory, remove_directory
warnings.filterwarnings("ignore")

st.title("RAG - AI 4 CI")

index_name = "docs-rag-summerday"
namespace = "summerday-space"
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = get_embeddings(embedding_model_name)
dimension=384 
metric="cosine"
spec=ServerlessSpec(cloud="aws", region="us-east-1")
db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)


################################################# Display vector data #####################################

if st.sidebar.button("View Data"):
  embeddings, chunks, vectors, vector_ids =  get_data_in_vector_store(index,namespace)
  documents_projected = create_2d_embeddings(embeddings)
  df = vectordb_to_dfdb(documents_projected, chunks, vectors,vector_ids)
  st.write(df)
  fig = df_visualisation(df)
  st.plotly_chart(fig)

############################################################################################################

############################################ upload files ###################################################
# File uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True)
# Define a mapping from file extensions to loader classes
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
# Define a mapping from file extensions to arguments loader
loader_kwargs_map = {
                    
              'pdf' : None,
              'txt' : None,
              'csv': {"csv_args": { "delimiter": ",","quotechar": '"'}},
              'json' : {'jq_schema':'.[] | "MainTask: \(.MainTask), MainTaskSummary: \(.MainTaskSummary), Tips: \(.Tips)  "' },
              'default': { 'strategy' :"fast"}
}

# Temporary directory where you want to save the files
directory = "./Internal_data"
create_directory(directory)
# Check if files were uploaded
if uploaded_files:
    save_uploaded_files(uploaded_files, directory)
    extensions_dict = organize_files_by_extension(directory)
    for ext, files in extensions_dict.items():
        # Create the glob pattern for the files of this extension
        glob_pattern = f'**/*.{ext}'
        loader_class = loader_cls_map.get(ext, loader_cls_map['default'])
        print("Extension: ",ext)
        print("Loader: ",loader_class)
        loader_args = loader_kwargs_map.get(ext, loader_kwargs_map['default'])
        documents = load_documents(directory, glob_pattern, loader_class, loader_args)
        chunks = split_documents(documents)
        db.add_documents(chunks)
        st.success(str(ext) + 'docs added successfully', icon="✅")
   
# Once processing is complete, remove the temporary directory and its contents
remove_directory(directory)
  
################################################################################################################################

# Configure your environment
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY'] 
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY'] 




chain_types = ['stuff',"refine", "map_reduce", "map_rerank"]
selected_chain_type = st.sidebar.selectbox("Choose a chain type:", options=chain_types)


# Sidebar for model and chain type selection
model_options = ["gemma-7b-it","mixtral-8x7b-32768","llama3-70b-8192","llama3-8b-8192"]
default_model = "llama3-70b-8192"  # Default model
selected_model = st.sidebar.selectbox("Choose a model:", options=model_options, index=model_options.index(default_model))





# Ensure we only initialize once and reinitialize if needed
if 'initialized' not in st.session_state or st.session_state.selected_chain_type != selected_chain_type or st.session_state.selected_model != selected_model:
    st.session_state.initialized = True
    st.session_state.selected_chain_type = selected_chain_type
    st.session_state.selected_model = selected_model

    ####

    map_reduce_question_template = """
    Use the following portion of a long document and the history of past interactions 
    to see if any of the text or previous context is relevant to answer the question. 
    Return any relevant text verbatim.

    History:
    --------
    {history}
    --------
    Context:
    --------
    {context}
    --------
    Question: {question}
    Relevant text, if any:
    """

    map_reduce_combine_template = """
    Given the following extracted parts of a long document, history of past interactions,
    and a question, create a final answer. If you don't know the answer, just say that you don't
    know. Don't try to make up an answer.

    History:
    --------
    {history}
    --------
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:
    """

    map_rerank_template = """
      Use the following pieces of context and history to answer the question at the end. 
      If you don't know the answer, just say that you don't know, don't try to make up an answer.

      In addition to giving an answer, also return a score of how fully it answered the user's question. 
      This should be in the following format:

      Question: [question here]
      Helpful Answer: [answer here]
      Score: [score between 0 and 100]

      How to determine the score:
      - Higher is a better answer
      - A better answer fully responds to the asked question, with sufficient level of detail
      - If you do not know the answer based on the context and history, that should be a score of 0
      - Don't be overconfident!

      Example #1

      History:
      --------
      None provided
      --------
      Context:
      --------
      Apples are red
      --------
      Question: What color are apples?
      Helpful Answer: Red
      Score: 100

      Example #2

      History:
      --------
      The car had been seen several times in the neighborhood.
      --------
      Context:
      --------
      It was night and the witness forgot his glasses. He was not sure if it was a sports car or an SUV.
      --------
      Question: What type was the car?
      Helpful Answer: A sports car or an SUV
      Score: 60

      Example #3

      History:
      --------
      Pears are a common topic in fruit studies.
      --------
      Context:
      --------
      Pears are either red or orange
      --------
      Question: What color are apples?
      Helpful Answer: This document does not answer the question
      Score: 0

      Begin!

      History:
      --------
      {history}
      --------
      Context:
      --------
      {context}
      --------
      Question: {question}
      Helpful Answer:
    """

    stuff_template = """
      Use the following pieces of history and context to answer the question at the end. 
      If you don't know the answer based on the provided information, just say that you don't know, don't try to make up an answer.
      History:
      --------
      {history}
      --------

      Context:
      --------
      {context}
      --------

      Question: {question}
      Helpful Answer:
      """
    refine_template ="""
        The original question is as follows: {question}
        We have provided an existing answer: {existing_answer}
        Here is the relevant history:
        ------------
        {history}
        ------------
        We have the opportunity to refine the existing answer (only if needed) with 
        some more context below.
        ------------
        {context_str}
        ------------
        Given the new context and history, refine the original answer to better answer the question. 
        If the context isn't useful, return the original answer.
        """
    refine_question_template = """
        Here is the relevant history:
        ------------
        {history}
        ------------
        Context information is below.
        ------------
        {context_str}
        ------------
        Given the context information, history, and no prior knowledge, answer the question: 
        {question}
            """

    #####

    ####
    chain_type_kwargs = {
      "map_reduce": {
          "question_prompt": PromptTemplate(input_variables=["context", "question", "history"], template=map_reduce_question_template),
          "combine_prompt": PromptTemplate(input_variables=["question", "summaries", "history"], template=map_reduce_combine_template),
          "memory": ConversationBufferMemory(memory_key="history", input_key="question")
      },
      "map_rerank": {
          "prompt": PromptTemplate(input_variables=["context", "question"], template= map_rerank_template,output_parser=RegexParser(regex=r"(.*?)\nScore: (\d*)",output_keys=["answer", "score"])),
          "memory": ConversationBufferMemory(memory_key="history", input_key="question")
      },
      "refine": {
          "refine_prompt": PromptTemplate(input_variables=["question", "existing_answer", "history", "context_str"], template=refine_template),
          "question_prompt": PromptTemplate(input_variables=["history", "context_str", "question"], template = refine_question_template),
          "memory": ConversationBufferMemory(memory_key="history", input_key="question")
      },
      "stuff": {
          "prompt": PromptTemplate(input_variables=["history", "context", "question"], template = stuff_template),
          "memory": ConversationBufferMemory(memory_key="history", input_key="question")
      }}
    ####
    
    
    # Setup Groq
    llm = ChatGroq(temperature=0, model_name=selected_model)
    
    # Define templates based on the selected chain type
    # Setup the retrieval_chain based on selected_chain_type
    args = chain_type_kwargs.get(selected_chain_type, chain_type_kwargs['stuff'])  # default to 'stuff' if not found
    st.session_state.retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type=selected_chain_type, retriever=db.as_retriever(search_type='mmr'), chain_type_kwargs=args)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if len(prompt.split()) > 500:
          st.warning(f"Your input is too long and will be truncated to fit the model's limit of 500 tokens.")
          prompt = ' '.join(prompt.split()[:500])  # Truncate the input
        response = st.session_state.retrieval_chain.run(prompt)
        matching_docs = db.as_retriever(search_type='mmr').get_relevant_documents(prompt)
        sources = [doc.metadata.get("source", doc.metadata) for doc in matching_docs]
        st.markdown(response)
        # Display sources
        if sources:
            st.markdown("### Sources:")
            for source in sources:
                st.markdown(f"- {source}")
    st.session_state.messages.append({"role": "assistant", "content": response})
# Sidebar Clear Chat Button
st.sidebar.button("Clear Chat", on_click=lambda: st.session_state.messages.clear())
