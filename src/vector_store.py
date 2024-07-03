from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os 


def get_vector_store(index_name, namespace, embeddings, dimension, metric, spec):
  # Setup Pinecone
  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
  if index_name not in pc.list_indexes().names():
      pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
  db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
  index = pc.Index(name=index_name)
  return db, index
