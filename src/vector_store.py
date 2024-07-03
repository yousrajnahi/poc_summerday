from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os 


def get_vector_store(index_name, namespace, embeddings, dimension, metric, spec):
  # Setup Pinecone
  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
  if index_name not in pc.list_indexes().names():
      pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
  db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
  index = pc.Index(name=index_name)
  return db, index



# Function to fetch and process data
def get_data_in_vector_store(index,namespace):
    id = []
    for ids in index.list(namespace=namespace):
        id.append(ids)
    all_ids = sum(id, [])
    vector_ids = [id for id in all_ids]
    response = index.fetch(ids=vector_ids, namespace=namespace)
    vectors = [vector_info['values'] for vector_info in response['vectors'].values()]
    chunks = [vector_info['metadata'] for vector_info in response['vectors'].values()]
    embeddings = np.array(vectors)
    return embeddings, chunks, vectors, vector_ids
