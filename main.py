import os
import streamlit as st
from dotenv import load_dotenv
import shutil

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage

from pages import PAGES

load_dotenv()

INDEX_DIR = "wiki_rag"

groq_api_key = os.getenv("GROQ_API_KEY")
Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")

@st.cache_resource
def get_index():
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")
    # Settings.llm = Ollama(model="llama3.2:latest", temperature=0)
    # Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.llm = Groq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    return index

@st.cache_resource
def get_query_engine():
    index = get_index()

    # llm = Ollama(model="llama3.2:latest", temperature=0)
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")
    llm = Groq(model="llama3-8b-8192", temperature=0,api_key=groq_api_key)
    Settings.llm = Groq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    return index.as_query_engine(llm=llm, similarity_top_k = 3)

def create_index():
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")
    Settings.llm = Groq(model="llama3-8b-8192", temperature=0, api_key=groq_api_key)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

def main():
    st.title("Wiki RAG App")

    if st.button("Actualizar índice"):
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.cache_resource.clear()
        index = create_index()
        st.success("Índice actualizado con éxito.")


    question = st.text_input("Ask Question")
    if st.button("Submit") and question:
        with st.spinner("Thinking..."):
            qa = get_query_engine()
            response = qa.query(question)

        st.subheader("Answer")
        st.write(response.response)

        st.subheader("Retrevid context")

        for src in response.source_nodes:
            st.markdown(src.node.get_content())


if __name__ == "__main__":
    main()