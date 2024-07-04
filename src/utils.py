import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from src.vector_store import get_vector_store
from src.embeddings import get_embeddings
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.regex import RegexParser
from src.document_processing import read_template_from_file

def initialize_vector_store():
    if 'vector_store' not in st.session_state:
        index_name = "docs-rag-summerday"
        namespace = "summerday-space"
        embedding_model_name = "all-MiniLM-L6-v2"
        embeddings = get_embeddings(embedding_model_name)
        dimension = 384
        metric = "cosine"
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)
        st.session_state.vector_store = db
        st.session_state.index = index
        st.session_state.namespace = "summerday-space"
    return st.session_state.vector_store, st.session_state.index, st.session_state.namespace


def get_or_create_retrieval_chain(selected_chain_type, selected_model, db):
    if 'retrieval_chain' not in st.session_state or \
       st.session_state.selected_chain_type != selected_chain_type or \
       st.session_state.selected_model != selected_model:
        
        # Load templates
        templates = {
            'map_reduce_question': read_template_from_file('prompt_templates/map_reduce_question_template.txt'),
            'map_reduce_combine': read_template_from_file('prompt_templates/map_reduce_combine_template.txt'),
            'map_rerank': read_template_from_file('prompt_templates/map_rerank_template.txt'),
            'stuff': read_template_from_file('prompt_templates/stuff_template.txt'),
            'refine': read_template_from_file('prompt_templates/refine_template.txt'),
            'refine_question': read_template_from_file('prompt_templates/refine_question_template.txt')
        }

        # Define chain type kwargs
        chain_type_kwargs = {
            "map_reduce": {
                "question_prompt": PromptTemplate(input_variables=["context", "question", "history"], template=templates['map_reduce_question']),
                "combine_prompt": PromptTemplate(input_variables=["question", "summaries", "history"], template=templates['map_reduce_combine']),
                "memory": ConversationBufferMemory(memory_key="history", input_key="question")
            },
            "map_rerank": {
                "prompt": PromptTemplate(input_variables=["context", "question"], template=templates['map_rerank'], 
                                         output_parser=RegexParser(regex=r"(.*?)\nScore: (\d*)", output_keys=["answer", "score"])),
                "memory": ConversationBufferMemory(memory_key="history", input_key="question")
            },
            "refine": {
                "refine_prompt": PromptTemplate(input_variables=["question", "existing_answer", "history", "context_str"], template=templates['refine']),
                "question_prompt": PromptTemplate(input_variables=["history", "context_str", "question"], template=templates['refine_question']),
                "memory": ConversationBufferMemory(memory_key="history", input_key="question")
            },
            "stuff": {
                "prompt": PromptTemplate(input_variables=["history", "context", "question"], template=templates['stuff']),
                "memory": ConversationBufferMemory(memory_key="history", input_key="question")
            }
        }   

        # Setup Groq
        llm = ChatGroq(temperature=0, model_name=selected_model)
        
        # Setup the retrieval_chain
        args = chain_type_kwargs.get(selected_chain_type, chain_type_kwargs['stuff'])
        retrieval_chain = RetrievalQA.from_chain_type(
            llm, 
            chain_type=selected_chain_type, 
            retriever=db.as_retriever(search_type='mmr'), 
            chain_type_kwargs=args
        )

        st.session_state.retrieval_chain = retrieval_chain
        st.session_state.selected_chain_type = selected_chain_type
        st.session_state.selected_model = selected_model
    
    return st.session_state.retrieval_chain
