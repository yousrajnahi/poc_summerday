import pacmap
import plotly.express as px
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from src.vector_store import get_vector_store, get_data_in_vector_store
from src.embeddings import get_embeddings
from pinecone import Pinecone, ServerlessSpec

# Function to create 2D embeddings
def create_2d_embeddings(embeddings):
    embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embeddings_2d = embedding_projector.fit_transform(embeddings, init="pca")
    return embeddings_2d

def vectordb_to_dfdb(documents_projected, chunks, vectors,vector_ids):
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
  return df

def df_visualisation(df):
  # Generating a custom color map
  num_unique_sources = df['source'].nunique()
  colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_unique_sources))
  color_discrete_map = {source: f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                      for source, rgb in zip(df['source'].unique(), colors)}
  # Plot
  fig = px.scatter(df, x="x", y="y", color="source", hover_data="extract", size="size_col", symbol="symbol", size_max=3, color_discrete_map=color_discrete_map, width=1000, height=700)
  fig.update_traces(marker=dict(opacity=0.7,line=dict(width=0.5), sizemode='diameter'), selector=dict(mode="markers"))
  fig.update_layout(legend_title_text="<b>Chunk source</b>", title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>")
  return fig

index_name = "docs-rag-summerday"
namespace = "summerday-space"
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = get_embeddings(embedding_model_name)
dimension = 384 
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)
embeddings, chunks, vectors, vector_ids =  get_data_in_vector_store(index,namespace)
documents_projected = create_2d_embeddings(embeddings)
df = vectordb_to_dfdb(documents_projected, chunks, vectors,vector_ids)
st.write(df)
fig = df_visualisation(df)
st.plotly_chart(fig)

