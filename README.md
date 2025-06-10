# RAG-Enriched Voice Agent

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- GROQ API key
- Google API key
- Cartesia API key
- LiveKit server

## Usage

1. Generate data from json file:
   ```bash
   python generate_fictional_store_data.py
   ```

2. Build the RAG database:
   ```bash
   python build_rag_data.py
   ```

3. Download model files:
   ```bash
   python main.py download-files
   ```

4. Run the agent:
   ```bash
   python main.py console
   ```