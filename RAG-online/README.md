# 🧠 RAG Ingestion Pipeline

This system allows ingestion of PDF documents via URLs and stores their chunked vector embeddings into a Qdrant collection, organized by user. It uses FastAPI as the API layer and LangChain + OpenAI + Qdrant under the hood for processing and storage.

---

## 📁 Project Structure

project/
├── rag_api.py # FastAPI endpoint for PDF ingestion
├── rag_builder.py # Embedding generation and Qdrant storage logic
├── .env # Stores environment variables for Qdrant

---

## 🔧 Environment Variables

Create a `.env` file in the root of your project with:

```env
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your-qdrant-api-key
```

---

## 📦 Requirements

Install the dependencies using:

pip install fastapi uvicorn pydantic python-dotenv requests \
            langchain langchain-community langchain-openai langchain-qdrant \
            qdrant-client