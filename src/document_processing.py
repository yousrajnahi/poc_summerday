__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
from langchain.output_parsers.regex import RegexParser
import warnings
import subprocess
import shutil
import pacmap
import plotly.express as px
import numpy as np
import pandas as pd


from langchain_community.document_loaders import PDFMinerLoader, DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
import os
import tempfile
from pdfminer.high_level import extract_text
from langchain_community.document_loaders import (UnstructuredFileLoader,
                                                  PDFMinerLoader,
                                                  CSVLoader,
                                                  JSONLoader,
                                                  TextLoader,
                                                  UnstructuredXMLLoader,
                                                  UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredEmailLoader)


import shutil
import matplotlib.pyplot as plt

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
def organize_files_by_extension(source_directory):
    # Initialize an empty dictionary to hold files organized by their extensions
    organized_dict = {}
    # Walk through all the directories and files in the source directory
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            # Get the full file path
            file_path = os.path.join(root, filename)
            # Split the file name to get the extension
            _, extension = os.path.splitext(filename)
            # Check if the file has an extension and is not a hidden file
            if extension and not filename.startswith('.'):
                # Normalize the extension by removing the leading dot and converting to lowercase
                extension_folder = extension[1:].lower()
                # If the extension is not already a key in the dictionary, add it with an empty list
                if extension_folder not in organized_dict:
                    organized_dict[extension_folder] = []
                # Append the file path to the list corresponding to its extension
                organized_dict[extension_folder].append(file_path)
    # Return the dictionary containing files organized by their extensions
    return organized_dict



def load_documents(DATA_PATH, glob_pattern,loader_class, loader_args):
  # Create a DirectoryLoader instance with specified parameters.
    loader = DirectoryLoader(DATA_PATH, glob=glob_pattern, show_progress=True,use_multithreading=True,loader_cls = loader_class, loader_kwargs =loader_args)
    # Load the documents using the configured loader and return them.
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

