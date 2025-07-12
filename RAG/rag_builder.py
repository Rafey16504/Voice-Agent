import argparse
import os
import uuid
import tempfile
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_url", type=str, required=True)
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    collection_name = f"user_{args.user_id}_embeddings"
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if args.reset:
        print(f"🧹 Resetting collection: {collection_name}")
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass  # Collection might not exist yet
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    documents = load_documents_from_url(args.pdf_url)
    chunks = split_documents(documents)

    if not chunks:
        raise Exception("❌ No valid chunks found. PDF may be empty or unreadable.")

    add_to_qdrant(client, chunks, collection_name)


def load_documents_from_url(pdf_url: str):
    print(f"⬇️ Downloading from: {pdf_url}")
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"❌ Failed to download PDF from {pdf_url}")

    print(f"✅ Downloaded {len(response.content)} bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    print(f"📄 Extracted {len(documents)} pages")

    if documents:
        print(f"🔎 Sample content:\n{documents[0].page_content[:300]}")

    return documents


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:2]):
        print(f"🧩 Chunk {i+1} preview:\n{chunk.page_content[:200]}\n")

    return chunks


def add_to_qdrant(client: QdrantClient, chunks: list[Document], collection_name: str):
    db = Qdrant(client=client, collection_name=collection_name, embeddings=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Sanity check embeddings
    print(f"🧠 Generating sample embedding...")
    sample_embedding = get_embedding_function().embed_query(chunks[0].page_content)
    print(f"✅ Embedding length: {len(sample_embedding)}")

    print(f"🚀 Adding {len(chunks_with_ids)} chunks to Qdrant...")
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)
    print("✅ Done!")


def calculate_chunk_ids(chunks: list[Document]):
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks


if __name__ == "__main__":
    main()
