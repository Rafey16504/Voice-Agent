# Import dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key) from .env file
load_dotenv()

# Alternative embedding options (currently commented out)
# from langchain_ollama import OllamaEmbeddings       # For using Ollama local embedding models
from langchain_openai import OpenAIEmbeddings         # For using OpenAI embedding models
# from langchain_community.embeddings.bedrock import BedrockEmbeddings  # For using Amazon Bedrock embeddings


def get_embedding_function():
    """
    Initializes and returns the embedding function for generating vector embeddings.

    Currently set to use OpenAI's 'text-embedding-3-small' model.
    Alternative embedding providers can be used by uncommenting relevant lines.
    """
    
    # Option 1: Amazon Bedrock Embeddings (Uncomment to use)
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default",   # AWS profile name from ~/.aws/credentials
    #     region_name="us-east-1"               # AWS region
    # )

    # Option 2: Ollama Local Embeddings (Uncomment to use)
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Option 3 (Active): OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return embeddings
