# rag_builder.py

import argparse
import os
import uuid
import tempfile
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

load_dotenv()

# Load Qdrant config from .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def load_documents_from_url(pdf_url: str) -> list[Document]:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"❌ Failed to download PDF from: {pdf_url}")

    print(f"✅ Downloaded {len(response.content)} bytes from {pdf_url}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    return loader.load()

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks

def add_to_qdrant(client: QdrantClient, chunks: list[Document], collection_name: str):
    db = Qdrant(client=client, collection_name=collection_name, embeddings=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)

    print(f"🧠 Adding {len(chunks_with_ids)} chunks to Qdrant collection: {collection_name}")
    ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=ids)

def main_from_api(pdf_urls: list[str], user_id: str, reset: bool = True):
    collection_name = f"user_{user_id}_embeddings"
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if reset:
        print(f"🗑️ Resetting collection: {collection_name}")
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    all_chunks = []

    for url in pdf_urls:
        print(f"📥 Processing PDF: {url}")
        documents = load_documents_from_url(url)
        if not documents:
            print(f"⚠️ No text extracted from {url}, skipping.")
            continue

        chunks = split_documents(documents)
        if not chunks:
            print(f"⚠️ No chunks from {url}, skipping.")
            continue

        all_chunks.extend(chunks)

    if not all_chunks:
        raise Exception("❌ No valid documents found from any provided PDFs.")

    add_to_qdrant(client, all_chunks, collection_name)

