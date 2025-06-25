import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession,
)
from livekit.plugins import google, silero, groq, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

CHROMA_PATH = str(Path(__file__).parent / "chroma")

class RAGEnrichedAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation) via Chroma DB.
    """

    def __init__(self) -> None:
        """Initialize the RAG-enabled agent."""
        super().__init__(
            instructions="""
You are a helpful voice assistant for the Orion Store. You can answer questions about the store’s departments, hours, services, staff/team, membership program, and upcoming events.
Keep your responses friendly, and conversational — like you're chatting with a neighbor. 
Avoid using technical jargon, markdown, or special formatting, and always speak clearly for text-to-speech output.
ALSO, DO NOT MENTION THAT YOU FOUND DATA FROM ANYWHERE, ANSWER AS IF YOU KNEW IT ALL ALREADY.
"""
        )

        try:
            self._embedding_function = get_embedding_function()
            self._db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self._embedding_function)
            logger.info("Chroma RAG database loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Chroma RAG database: {e}")

        self._seen_results = set()  # Track previously seen results

    @function_tool
    async def orion_info_search(self, context: RunContext, query: str):
        """Search Orion Store’s knowledge base for helpful information. Logs latency of query processing."""
        start_time = time.time()
        try:
            results = self._db.similarity_search_with_score(query, k=5)

            new_results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("id") not in self._seen_results
            ]

            if len(new_results) == 0:
                logger.info(f"[Latency: {time.time() - start_time:.2f}s] No new results for query: '{query}'")
                return "I couldn’t find any new information about that. Try rephrasing your question?"

            new_results = new_results[:2]

            context_parts = []
            for doc, _score in new_results:
                chunk_id = doc.metadata.get("id", "Unknown source")
                content = doc.page_content.strip()
                self._seen_results.add(chunk_id)
                context_parts.append(f"Source: {chunk_id}\nContent: {content}\n")

            full_context = "\n\n".join(context_parts)
            escaped_context = full_context.replace("\n", "\\n")

            latency = time.time() - start_time
            logger.info(f"[Latency: {latency:.2f}s] Query: '{query}' | Results: {len(new_results)} | Context: {escaped_context}")

            return full_context

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return "Sorry, something went wrong while looking that up."

    async def on_enter(self):
        """Greet the user when the session begins."""
        self.session.generate_reply(
            instructions="GREET THE USER AT THE START OF THE CONVERSATION"
        )

    async def on_transcription(self, text: str):
        start_time = time.time()

        cleaned_text = text.strip().lower()

        # Skip empty or very short transcriptions
        if len(cleaned_text) < 3:
            logger.info("Skipping short or empty transcription.")
            return

        # Skip common filler phrases
        ignored_phrases = {"uh", "um", "hmm", "okay", "ok", "huh", "hmm okay"}
        if cleaned_text in ignored_phrases:
            logger.info(f"Skipping filler phrase: '{cleaned_text}'")
            return

        await self.session.generate_reply(prompt=text)
        latency = time.time() - start_time
        logger.info(f"[Full Response Latency: {latency:.2f}s] Prompt: {text}")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    await ctx.connect()

    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
        tts=cartesia.TTS(),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))