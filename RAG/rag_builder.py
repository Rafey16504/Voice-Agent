import os
import uuid
import tempfile
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings  
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def main_from_api(pdf_url: str, user_id: str):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = f"user_{user_id}_embeddings"

    # Reset collection
    try:
        client.delete_collection(collection_name=collection_name)
    except:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )

    documents = load_documents_from_url(pdf_url)
    chunks = split_documents(documents)
    add_to_qdrant(client, chunks, collection_name)

def load_documents_from_url(pdf_url: str):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF from {pdf_url}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    return loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def add_to_qdrant(client: QdrantClient, chunks: list[Document], collection_name: str):
    db = Qdrant(client=client, collection_name=collection_name, embeddings=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"👉 Adding {len(chunks_with_ids)} chunks to Qdrant")
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)

def calculate_chunk_ids(chunks):
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks
