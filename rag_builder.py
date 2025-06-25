import argparse
import os
import shutil
import uuid
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "orion_store_embeddings"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the Qdrant collection.")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if args.reset:
        print("✨ Clearing Qdrant Collection")
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
        except:
            pass  # Ignore if it doesn't exist
        print(f"✅ Creating Qdrant collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_qdrant(client, chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_qdrant(client: QdrantClient, chunks: list[Document]):
    db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=get_embedding_function())


    chunks_with_ids = calculate_chunk_ids(chunks)

    print(f"👉 Adding {len(chunks_with_ids)} chunks to Qdrant")
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)


def calculate_chunk_ids(chunks):
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks


if __name__ == "__main__":
    main()