# RAG-Enriched Voice Agent

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- GROQ API key
- Google API key
- Cartesia API key
- LiveKit server
- Qdrant URL and API key


### 1. Build the RAG Database (Qdrant)
Run the following to process documents, generate embeddings, and store them on Qdrant:
```bash
python rag_builder.py --reset
```

### 2. Run the Voice Agent in Terminal (Console Mode)
This runs the voice agent directly in your terminal for testing:
```bash
python main.py console
```

### 3. Connect the Voice Agent to a LiveKit Room
To connect the agent to a specific LiveKit room:
```bash
python main.py connect --room <room_name>
```