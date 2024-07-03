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
import shutil
import matplotlib.pyplot as plt
import nltk

def get_vector_store(index_name, namespace):
  # Setup Pinecone
  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
  #index_name = "docs-rag-summerday"
  if index_name not in pc.list_indexes().names():
      pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
  #namespace = "summerday-space"
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
  index = pc.Index(name=index_name)
  return embeddings, db, index
