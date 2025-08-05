import os
import shutil
import pandas as pd
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "wiki_rag"
CSV_FILE = "pages.csv"

Settings.embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
Settings.llm = Groq(model="llama3-8b-8192", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

def load_pages():
    if not os.path.exists(CSV_FILE):
        return []
    return pd.read_csv(CSV_FILE)["page"].tolist()

def save_pages(pages):
    pd.DataFrame({"page": pages}).to_csv(CSV_FILE, index=False)

def add_page(new_page):
    pages = load_pages()
    if new_page not in pages:
        pages.append(new_page)
        save_pages(pages)
        rebuild_index(pages)
        return True
    return False

def delete_page(page):
    pages = load_pages()
    if page in pages:
        pages.remove(page)
        save_pages(pages)
        rebuild_index(pages)
        return True
    return False

def rebuild_index(pages=None):
    if pages is None:
        pages = load_pages()
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    docs = WikipediaReader().load_data(pages=pages, auto_suggest=False)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)

def load_query_engine():
    if os.path.exists(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage)
    else:
        pages = load_pages()
        docs = WikipediaReader().load_data(pages=pages, auto_suggest=False)
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    return index.as_query_engine(similarity_top_k=3)
