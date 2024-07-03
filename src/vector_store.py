from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os 
from src.embeddings import get_embeddings

def get_vector_store(index_name, namespace):
  # Setup Pinecone
  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
  if index_name not in pc.list_indexes().names():
      pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
  #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  embeddings = get_embeddings("all-MiniLM-L6-v2")
  db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
  index = pc.Index(name=index_name)
  return embeddings, db, index
