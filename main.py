# Import standard libraries
import logging  # For logging information, warnings, errors
import time  # For measuring latency
from pathlib import Path  # For working with file paths
from dotenv import load_dotenv  # To load environment variables from a .env file
import os  # For interacting with environment variables

# Import LiveKit Agents SDK components
from livekit.agents import (
    JobContext,         # Context for the running agent job
    WorkerOptions,      # Options for the agent worker process
    cli,                # CLI utilities to run the app
    RunContext,         # Context for specific function tools
    function_tool,      # Decorator to expose functions to the agent
    RoomInputOptions,   # Options for handling room input
    Agent,              # Base class for building agents
    AgentSession,       # Manages the agent's session lifecycle
)

# Import plugins for LLM, TTS, STT, VAD
from livekit.plugins import google, silero, groq, cartesia, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # Optional VAD model (currently unused)

# Import RAG and Qdrant components
from get_embedding_function import get_embedding_function  # Your custom embedding function
from langchain_qdrant import Qdrant  # Wrapper for Qdrant integration with LangChain
from qdrant_client import QdrantClient  # Qdrant client for direct database operations

# Load environment variables from .env file
load_dotenv()

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

# Load Qdrant configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "user_1585bdff-6424-416b-9a51-12908fb5c799_embeddings"  # Name of your Qdrant collection

# Define the RAG-enabled Agent class
class RAGEnrichedAgent(Agent):
    """
    Custom voice assistant for the Orion Store with Retrieval Augmented Generation (RAG).
    Searches a Qdrant database for knowledge and answers questions conversationally.
    """

    def __init__(self) -> None:
        """Initialize the RAG-enabled agent with instructions and RAG setup."""
        super().__init__(
            instructions="""
You are a helpful voice assistant for the user. You can answer questions about the file they have uploaded.
Keep your responses friendly, and conversational — like you're chatting with a neighbor. 
Avoid using technical jargon, markdown, or special formatting, and always speak clearly for text-to-speech output.
ALSO, DO NOT MENTION THAT YOU FOUND DATA FROM ANYWHERE, ANSWER AS IF YOU KNEW IT ALL ALREADY.
"""
        )

        # Try to initialize Qdrant RAG database
        try:
            self._embedding_function = get_embedding_function()  # Your embedding function for queries
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)  # Connect to Qdrant
            self._db = Qdrant(
                client=client,
                collection_name=COLLECTION_NAME,
                embeddings=self._embedding_function
            )
            logger.info("Qdrant RAG database loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qdrant RAG database: {e}")

        self._seen_results = set()  # Keeps track of previously seen chunks to avoid repeating them

    @function_tool
    async def info_search(self, context: RunContext, query: str):
        """
        Function tool to search Orion Store’s knowledge base using RAG.
        Avoids repeating previously seen results and logs search latency.
        """
        start_time = time.time()
        try:
            results = self._db.similarity_search_with_score(query, k=2)  # Top 5 similar chunks

            # Filter out previously seen results
            new_results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("id") not in self._seen_results
            ]

            if len(new_results) == 0:
                logger.info(f"[Latency: {time.time() - start_time:.2f}s] No new results for query: '{query}'")
                return "I couldn’t find any new information about that. Try rephrasing your question?"

            # Limit to top 2 new results to avoid overloading response
            new_results = new_results[:2]

            # Prepare the final response context
            context_parts = []
            for doc, _score in new_results:
                chunk_id = doc.metadata.get("id", "Unknown source")
                content = doc.page_content.strip()
                self._seen_results.add(chunk_id)
                context_parts.append(f"Source: {chunk_id}\nContent: {content}\n")

            # Join results and escape newlines for clean logging
            full_context = "\n\n".join(context_parts)
            escaped_context = full_context.replace("\n", "\\n")

            latency = time.time() - start_time
            logger.info(f"[Latency: {latency:.2f}s] Query: '{query}' | Results: {len(new_results)} | Context: {escaped_context}")

            return full_context

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return "Sorry, something went wrong while looking that up."

    async def on_enter(self):
        """
        Event hook triggered when the user joins the session.
        Sends a greeting automatically.
        """
        self.session.generate_reply(
            instructions="GREET THE USER AT THE START OF THE CONVERSATION"
        )

    async def on_transcription(self, text: str):
        """
        Event hook triggered when user speech is transcribed.
        Filters short/filler phrases, otherwise generates a reply.
        """
        start_time = time.time()

        cleaned_text = text.strip().lower()

        # Skip empty or too short transcriptions
        if len(cleaned_text) < 3:
            logger.info("Skipping short or empty transcription.")
            return

        # Common filler phrases to ignore
        ignored_phrases = {"uh", "um", "hmm", "okay", "ok", "huh", "hmm okay"}
        if cleaned_text in ignored_phrases:
            logger.info(f"Skipping filler phrase: '{cleaned_text}'")
            return

        # Generate a reply to valid input
        await self.session.generate_reply(prompt=text)

        latency = time.time() - start_time
        logger.info(f"[Full Response Latency: {latency:.2f}s] Prompt: {text}")

# Main entrypoint function for the agent worker
async def entrypoint(ctx: JobContext):
    """
    Main function for starting the agent.
    Connects to LiveKit, sets up session with LLM, and starts the RAG-enabled agent.
    """
    await ctx.connect()

    # You can swap this to a different LLM or pipeline structure if needed
    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-live-001",  # Fast Gemini-2.0 model
            voice="Kore",  # Voice for TTS
            temperature=0.8  # Controls randomness/creativity of responses
        ),
    )

    # Start the session with your custom RAG agent
    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

# Runs the agent app when the file is executed
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
