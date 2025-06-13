# RAG-Enriched Voice Agent

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- GROQ API key
- Google API key
- Cartesia API key
- LiveKit server

## Usage

1. Enter data in json file:

2. Build the RAG database:
   ```bash
   python rag_builder.py
   ```

3. Download model files:
   ```bash
   python main.py download-files
   ```

4. Run the agent:
   ```bash
   python main.py console
   ```