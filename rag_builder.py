# rag_builder.py
import asyncio
import logging
import json
import uuid
import pickle
from pathlib import Path
from typing import List, Union, Optional, Any
from dataclasses import dataclass

import aiohttp
from tqdm import tqdm
from dotenv import load_dotenv
from livekit.plugins import openai
from livekit.agents import tokenize

from rag_index import IndexBuilder, SentenceChunker

load_dotenv()
logger = logging.getLogger("rag-builder")

def bullet_point_chunker(text: str) -> List[str]:
        # This assumes each bullet starts with a dash, possibly with a space
        return [line.strip() for line in text.split('\n') if line.strip().startswith("-")]
        
class RAGBuilder:
    def __init__(
        self,
        index_path: Union[str, Path],
        data_path: Union[str, Path],
        embeddings_dimension: int = 1536,
        embeddings_model: str = "text-embedding-3-small",
        metric: str = "angular",
    ):
        self._index_path = Path(index_path)
        self._data_path = Path(data_path)
        self._embeddings_dimension = embeddings_dimension
        self._embeddings_model = embeddings_model
        self._metric = metric

    def _extract_paragraphs_from_json(self, obj: Any, path: str = "") -> List[str]:
        paragraphs = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                paragraphs.extend(self._extract_paragraphs_from_json(v, f"{path}.{k}" if path else k))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                paragraphs.extend(self._extract_paragraphs_from_json(item, f"{path}[{i}]"))
        elif isinstance(obj, str) and obj.strip():
            paragraphs.append(obj.strip())
        return paragraphs

    def _clean_content(self, text: str) -> str:
        skip_patterns = [
            'Docs', 'Search', 'GitHub', 'Slack', 'Sign in', 'Home', 'AI Agents',
            'Telephony', 'Recipes', 'Reference', 'On this page',
            'Get started with LiveKit today', 'Content from https://docs.livekit.io/'
        ]
        return '\n'.join(
            line.strip()
            for line in text.split('\n')
            if line.strip() and not any(p in line for p in skip_patterns) and not line.startswith('http')
        )

    

    async def _create_embeddings(self, text: str, http_session: Optional[aiohttp.ClientSession]) -> openai.EmbeddingData:
        results = await openai.create_embeddings(
            input=[text],
            model=self._embeddings_model,
            dimensions=self._embeddings_dimension,
            http_session=http_session,
        )
        return results[0]

    async def build_from_texts(self, texts: List[str], show_progress: bool = True) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as http_session:
            idx_builder = IndexBuilder(self._embeddings_dimension, self._metric)
            cleaned = [self._clean_content(t) for t in texts if t.strip()]
            paragraphs_by_uuid = {str(uuid.uuid4()): t for t in cleaned}

            items = tqdm(paragraphs_by_uuid.items(), desc="Creating embeddings") if show_progress else paragraphs_by_uuid.items()
            for p_uuid, paragraph in items:
                embedding = await self._create_embeddings(paragraph, http_session)
                idx_builder.add_item(embedding.embedding, p_uuid)

            logger.info("Building index...")
            idx_builder.build()
            idx_builder.save(str(self._index_path))

            with open(self._data_path, "wb") as f:
                pickle.dump(paragraphs_by_uuid, f)

    async def build_from_json_file(self, file_path: Union[str, Path], show_progress: bool = True) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        raw_paragraphs = self._extract_paragraphs_from_json(json_data)
        chunker = SentenceChunker()
        all_chunks = []
        for para in raw_paragraphs:
            cleaned = self._clean_content(para)
            if "-" in cleaned:
                chunks = bullet_point_chunker(cleaned)
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    all_chunks.append(cleaned)
            else:
                all_chunks.extend(chunker.chunk(text=cleaned))
        await self.build_from_texts(all_chunks, show_progress)

async def main() -> None:
    json_data_path = Path(__file__).parent / "orion_store.json"
    if not json_data_path.exists():
        logger.error("store_data.json not found. Run generate_fictional_store_data.py first.")
        return

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    builder = RAGBuilder(
        index_path=output_dir,
        data_path=output_dir / "paragraphs.pkl",
        embeddings_dimension=1536,
    )
    logger.info("Building RAG database from JSON...")
    await builder.build_from_json_file(file_path=json_data_path)
    logger.info("RAG database built and saved.")

if __name__ == "__main__":
    asyncio.run(main())
