import logging
import pickle
import random
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Any, Literal
from collections.abc import Iterable
from dataclasses import dataclass

import annoy

from livekit.agents.voice import Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai

logger = logging.getLogger("rag-handler")

# Constants
ANNOY_FILE = "index.annoy"
METADATA_FILE = "metadata.pkl"
Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]

# Data classes
@dataclass
class Item:
    i: int
    userdata: Any
    vector: list[float]

@dataclass
class _FileData:
    f: int  # dimensions
    metric: Metric
    userdata: dict[int, Any]

@dataclass
class QueryResult:
    userdata: Any
    distance: float

# Annoy index wrapper
class AnnoyIndex:
    def __init__(self, index: annoy.AnnoyIndex, filedata: _FileData) -> None:
        self._index = index
        self._filedata = filedata

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnnoyIndex":
        p = Path(path)
        with open(p / METADATA_FILE, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(p / ANNOY_FILE))
        return cls(index, metadata)

    def query(self, vector: list[float], n: int, search_k: int = -1) -> list[QueryResult]:
        ids, distances = self._index.get_nns_by_vector(vector, n, search_k=search_k, include_distances=True)
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=dist)
            for i, dist in zip(ids, distances)
        ]

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            yield Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i)
            )

# Thinking behavior enum
class ThinkingStyle(Enum):
    NONE = "none"
    MESSAGE = "message"
    LLM = "llm"

DEFAULT_THINKING_MESSAGES = [
    "Let me look that up...",
    "One moment while I check...",
    "I'll find that information for you...",
    "Just a second while I search...",
    "Looking into that now..."
]
DEFAULT_THINKING_PROMPT = "Generate a very short message to indicate that we're looking up the answer in the docs."

# RAG handler class
class RAGHandler:
    def __init__(
        self,
        index_path: Union[str, Path],
        data_path: Union[str, Path],
        thinking_style: Union[str, ThinkingStyle] = ThinkingStyle.MESSAGE,
        thinking_messages: Optional[List[str]] = None,
        thinking_prompt: Optional[str] = None,
        embeddings_dimension: int = 1536,
        embeddings_model: str = "text-embedding-3-small"
    ):
        self._index_path = Path(index_path)
        self._data_path = Path(data_path)
        self._thinking_style = ThinkingStyle(thinking_style) if isinstance(thinking_style, str) else thinking_style
        self._thinking_messages = thinking_messages or DEFAULT_THINKING_MESSAGES
        self._thinking_prompt = thinking_prompt or DEFAULT_THINKING_PROMPT
        self._embeddings_dimension = embeddings_dimension
        self._embeddings_model = embeddings_model

        if not self._index_path.exists():
            raise FileNotFoundError(f"Annoy index path not found: {self._index_path}")
        if not self._data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self._data_path}")

        self._annoy_index = AnnoyIndex.load(self._index_path)
        with open(self._data_path, "rb") as f:
            self._paragraphs_by_uuid = pickle.load(f)

    async def _handle_thinking(self, agent: Agent) -> None:
        if self._thinking_style == ThinkingStyle.NONE:
            return
        elif self._thinking_style == ThinkingStyle.MESSAGE:
            await agent.session.say(random.choice(self._thinking_messages))
        elif self._thinking_style == ThinkingStyle.LLM:
            response = await agent._llm.complete(self._thinking_prompt)
            await agent.session.say(response.text)

    async def retrieve_context(self, query: str) -> str:
        embeddings = await openai.create_embeddings(
            input=[query],
            model=self._embeddings_model,
            dimensions=self._embeddings_dimension
        )
        query_vector = embeddings[0].embedding
        results = self._annoy_index.query(query_vector, n=1)
        if not results:
            return ""
        return self._paragraphs_by_uuid.get(results[0].userdata, "")

    async def enrich_with_rag(self, agent: Agent, context: RunContext, query: str) -> None:
        await self._handle_thinking(agent)
        relevant_context = await self.retrieve_context(query)

        if not relevant_context:
            await agent.session.say("I couldn't find any relevant information about that.")
            return

        prompt = f"""
        Question: {query}

        Relevant information:
        {relevant_context}

        Based on the above information, give a helpful and concise answer to the question.
        """
        response = await agent._llm.complete(prompt)
        await agent.session.say(response.text)

    def register_with_agent(self, agent: Agent) -> None:
        @function_tool
        async def lookup_info(self, context: RunContext, query: str):
            logger.info(f"Looking up info for query: {query}")
            await self.rag_handler.enrich_with_rag(self, context, query)

        agent.lookup_info = lookup_info.__get__(agent)
        agent.rag_handler = self
