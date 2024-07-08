import pacmap
import plotly.express as px
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler



def create_2d_embeddings(embeddings):
    # Perform t-SNE
    np.random.seed(42)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Normalize the embeddings
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #embeddings_2d_normalized = scaler.fit_transform(embeddings_2d)
    return embeddings_2d

def vectordb_to_dfdb(documents_projected, chunks, vectors, vector_ids, matching_docs=None):
    source_matching_docs = []
    text_matching_docs = []
    
    if matching_docs:
        source_matching_docs = [doc.metadata['source'] for doc in matching_docs]
        text_matching_docs = [doc.page_content for doc in matching_docs]
    
    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": chunks[i]["source"].split("/")[-1],
                "extract": chunks[i]['text'][:100] + "...",
                "size_col": 4,
                "symbol": "star" if (matching_docs and 
                                     chunks[i]["source"] in source_matching_docs and 
                                     chunks[i]['text'] in text_matching_docs) else "circle",
                'vector': vectors[i],
                'id': vector_ids[i]
            }
            for i in range(len(chunks))
        ]
    )
    return df

def df_visualisation(df):
  num_unique_sources = df['source'].nunique()
  colors = plt.get_cmap('tab20')(np.linspace(0, 1, num_unique_sources))
  color_discrete_map = {source: f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                      for source, rgb in zip(df['source'].unique(), colors)}
  # Ensure user query has a distinct color
  color_discrete_map["User query"] = "black"
 
  # Plot
  fig = px.scatter(df, x="x", y="y", color="source", size="size_col", symbol="symbol", size_max=3,color_discrete_map = color_discrete_map, width=1000, height=700)
  fig.update_traces(marker=dict(opacity=0.7,line=dict(width=0.5), sizemode='diameter'), selector=dict(mode="markers"))
  fig.update_layout(legend_title_text="<b>Chunk source</b>", title="<b>2D Projection of Chunk Embeddings via TSNE</b>")
  return fig

# Extract file extensions
def get_file_extension(filename):
    return filename.split('.')[-1] if filename else ''
