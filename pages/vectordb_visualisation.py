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

from src.vector_store import get_vector_store, get_data_in_vector_store
from src.embeddings import get_embeddings

# Function to create 2D embeddings
def create_2d_embeddings(embeddings):
    embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embeddings_2d = embedding_projector.fit_transform(embeddings, init="pca")
    return embeddings_2d

embeddings, chunks, vectors, vector_ids =  get_data_in_vector_store(index,namespace)
documents_projected = create_2d_embeddings(embeddings)

# Create DataFrame
df = pd.DataFrame.from_dict(
  [
      {
          "x": documents_projected[i, 0],
          "y": documents_projected[i, 1],
          "source": chunks[i]["source"].split("/")[-1],
          "extract": chunks[i]['text'][:100] + "...",
          "symbol": "circle",
          "size_col": 4,  # Reduced size
          'vector': vectors[i],
          'id': vector_ids[i]
      }
      for i in range(len(chunks))
  ]
)
st.write(df)

# Generating a custom color map
num_unique_sources = df['source'].nunique()
colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_unique_sources))

color_discrete_map = {source: f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                    for source, rgb in zip(df['source'].unique(), colors)}

# Ensure user query has a distinct color
color_discrete_map["User query"] = "black"

# Plot
fig = px.scatter(
  df,
  x="x",
  y="y",
  color="source",
  hover_data="extract",
  size="size_col",
  symbol="symbol",
  size_max=3,
  color_discrete_map=color_discrete_map,
  width=1000,
  height=700
)

fig.update_traces(
  marker=dict(
      opacity=0.7,
      line=dict(width=0.5),
      sizemode='diameter'
  ),
  selector=dict(mode="markers")
)

fig.update_layout(
  legend_title_text="<b>Chunk source</b>",
  title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>"
)

st.plotly_chart(fig)
