import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.regex import RegexParser
from src.vector_store import get_vector_store
from src.embeddings import get_embeddings
from src.document_processing import read_template_from_file

st.title("Chat")

# Configure your environment
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY'] 
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY'] 

# Sidebar for model and chain type selection
chain_types = ['stuff', "refine", "map_reduce", "map_rerank"]
selected_chain_type = st.sidebar.selectbox("Choose a chain type:", options=chain_types)

model_options = ["gemma-7b-it", "mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"]
default_model = "llama3-70b-8192"
selected_model = st.sidebar.selectbox("Choose a model:", options=model_options, index=model_options.index(default_model))

# Initialize vector store and embeddings
index_name = "docs-rag-summerday"
namespace = "summerday-space"
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = get_embeddings(embedding_model_name)
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
db, index = get_vector_store(index_name, namespace, embeddings, dimension, metric, spec)

# Initialize retrieval chain
if 'retrieval_chain' not in st.session_state or st.session_state.selected_chain_type != selected_chain_type or st.session_state.selected_model != selected_model:
    st.session_state.selected_chain_type = selected_chain_type
    st.session_state.selected_model = selected_model

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
    st.session_state.retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type=selected_chain_type, retriever=db.as_retriever(search_type='mmr'), chain_type_kwargs=args)

# Chat interface
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
            prompt = ' '.join(prompt.split()[:500])
        response = st.session_state.retrieval_chain.run(prompt)
        matching_docs = db.as_retriever(search_type='mmr').get_relevant_documents(prompt)
        sources = [doc.metadata.get("source", doc.metadata) for doc in matching_docs]
        st.markdown(response)
        if sources:
            st.markdown("### Sources:")
            for source in sources:
                st.markdown(f"- {source}")
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar Clear Chat Button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages.clear()
