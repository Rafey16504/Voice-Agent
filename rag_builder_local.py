# Import standard libraries
import argparse  # For parsing command-line arguments
import os  # For environment variable access
import shutil  # (Unused but imported, can be removed if not needed)
import uuid  # For generating unique IDs for document chunks

# Import LangChain document loader and text splitter
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Loads PDFs from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits documents into manageable chunks
from langchain.schema.document import Document  # Represents a document structure

# Import your embedding function and Qdrant integration
from get_embedding_function import get_embedding_function  # Your custom embedding function for vector generation
from langchain_qdrant import Qdrant  # Wrapper for Qdrant with LangChain compatibility
from qdrant_client import QdrantClient, models  # Qdrant client for direct operations and model definitions

# Load Qdrant configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "orion_store_embeddings"  # Name of the collection to store embeddings
DATA_PATH = "data"  # Path to directory containing PDF files


def main():
    """
    Main entrypoint for indexing documents into Qdrant.
    Supports an optional --reset flag to clear and recreate the Qdrant collection.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the Qdrant collection.")
    args = parser.parse_args()

    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if args.reset:
        print("✨ Clearing Qdrant Collection")
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)  # Delete existing collection if present
        except:
            pass  # Ignore error if collection doesn't exist
        print(f"✅ Creating Qdrant collection '{COLLECTION_NAME}'")
        
        # Create new collection with vector size and distance metric
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    # Load, split, and index documents
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_qdrant(client, chunks)


def load_documents():
    """
    Loads all PDF files from the specified directory as LangChain documents.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    Splits loaded documents into smaller chunks for better embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # Max characters per chunk
        chunk_overlap=30,      # Overlap between chunks to preserve context
        length_function=len,   # Use Python's len() to measure chunk size
        is_separator_regex=False,  # Use plain character-based splitting
    )
    return text_splitter.split_documents(documents)


def add_to_qdrant(client: QdrantClient, chunks: list[Document]):
    """
    Adds document chunks to the Qdrant collection after generating unique IDs.
    """
    db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=get_embedding_function())

    chunks_with_ids = calculate_chunk_ids(chunks)

    print(f"👉 Adding {len(chunks_with_ids)} chunks to Qdrant")
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)


def calculate_chunk_ids(chunks):
    """
    Adds a unique UUID to each chunk's metadata to serve as its identifier.
    """
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks


if __name__ == "__main__":
    main()
