
"""
This agent uses the RAG (Retrieval Augmented Generation) plugin to provide
information from a knowledge base when answering user questions.

Before running this agent:
1. Make sure you have your GROQ, Cartesia, OpenAI and Google API key in a .env file
2. Run build_rag_data.py to build the RAG database
"""
import logging
import pickle
from pathlib import Path
from typing import Literal, Any
from collections.abc import Iterable
from dataclasses import dataclass
from dotenv import load_dotenv
import annoy
import time

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
from livekit.plugins import google, openai, silero, groq, cartesia, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

# RAG Index Types and Classes
Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]
ANNOY_FILE = "index.annoy"
METADATA_FILE = "metadata.pkl"


@dataclass
class _FileData:
    f: int
    metric: Metric
    userdata: dict[int, Any]


@dataclass
class Item:
    i: int
    userdata: Any
    vector: list[float]


@dataclass
class QueryResult:
    userdata: Any
    distance: float


class AnnoyIndex:
    def __init__(self, index: annoy.AnnoyIndex, filedata: _FileData) -> None:
        self._index = index
        self._filedata = filedata

    @classmethod
    def load(cls, path: str) -> "AnnoyIndex":
        p = Path(path)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE

        with open(metadata_path, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(index_path))
        return cls(index, metadata)

    @property
    def size(self) -> int:
        return self._index.get_n_items()

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            item = Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i),
            )
            yield item

    def query(
        self, vector: list[float], n: int, search_k: int = -1
    ) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(
            vector, n, search_k=search_k, include_distances=True
        )
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=distance)
            for i, distance in zip(*ids)
        ]


class RAGEnrichedAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation).
    """

    def __init__(self) -> None:
        """Initialize the RAG-enabled agent."""
        super().__init__(
            instructions = """
You are a helpful voice assistant for the Orion General Store, a community-focused general store in Redwood Valley.
You can answer questions about the store’s departments, hours, services, staff, membership program, and upcoming events.
Keep your responses friendly, and conversational — like you're chatting with a neighbor. 
Avoid using technical jargon, markdown, or special formatting, and always speak clearly for text-to-speech output.
GREET THE USER AT THE START. ALSO, DO NOT MENTION THAT YOU FOUND DATA FROM ANYWHERE, ANSWER AS IF YOU KNEW IT ALL ALREADY.
"""

        )

        # Initialize RAG components
        vdb_dir = Path(__file__).parent / "data"
        data_path = vdb_dir / "paragraphs.pkl"

        if not vdb_dir.exists() or not data_path.exists():
            logger.warning(
                "RAG database not found. Please run build_rag_data.py first:\n"
                "$ python build_rag_data.py"
            )
            return

        # Load RAG index and data
        self._index_path = vdb_dir
        self._data_path = data_path
        self._embeddings_dimension = 1536
        self._embeddings_model = "text-embedding-3-small"
        self._seen_results = set()  # Track previously seen results

        try:
            self._annoy_index = AnnoyIndex.load(str(self._index_path))
            with open(self._data_path, "rb") as f:
                self._paragraphs_by_uuid = pickle.load(f)
            logger.info("RAG database loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RAG database: {e}")

    @function_tool
    async def orion_info_search(self, context: RunContext, query: str):
        """Search Orion General Store’s knowledge base for helpful information. Logs latency of query processing."""
        start_time = time.time()  # Start timing
        try:
            # Generate embeddings for the query
            query_embedding = await openai.create_embeddings(
                input=[query],
                model=self._embeddings_model,
                dimensions=self._embeddings_dimension,
            )

            all_results = self._annoy_index.query(query_embedding[0].embedding, n=5)

            # Filter out previously seen results
            new_results = [
                r for r in all_results if r.userdata not in self._seen_results
            ]

            if len(new_results) == 0:
                logger.info(f"[Latency: {time.time() - start_time:.2f}s] No new results for query: '{query}'")
                return "I couldn’t find any new information about that. Try rephrasing your question?"

            new_results = new_results[:2]  # Limit to 2 top new results

            context_parts = []
            for result in new_results:
                self._seen_results.add(result.userdata)
                paragraph = self._paragraphs_by_uuid.get(result.userdata, "")
                if paragraph:
                    source = "Unknown source"
                    if "from [" in paragraph:
                        source = paragraph.split("from [")[1].split("]")[0]
                        paragraph = paragraph.split("]")[1].strip()
                    context_parts.append(f"Source: {source}\nContent: {paragraph}\n")

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
            instructions="Hi there! Welcome to the Orion General Store. How can I help you today?"
        )



async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    await ctx.connect()

    
    session = AgentSession(
        llm = google.beta.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",
        voice="Kore",
        temperature=0.8
    ),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))