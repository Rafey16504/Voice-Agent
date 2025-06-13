import pickle
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Union, Literal, Callable, Any
from collections.abc import Iterable
from dataclasses import dataclass
import aiohttp
from tqdm import tqdm
import annoy
import asyncio
from livekit.agents import tokenize
from livekit.plugins import openai
import json

load_dotenv()
logger = logging.getLogger("rag-builder")

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


class IndexBuilder:
    def __init__(self, f: int, metric: Metric) -> None:
        self._index = annoy.AnnoyIndex(f, metric)
        self._filedata = _FileData(f=f, metric=metric, userdata={})
        self._i = 0

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE
        self._index.save(str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self._filedata, f)

    def build(self, trees: int = 50, jobs: int = -1) -> AnnoyIndex:
        self._index.build(n_trees=trees, n_jobs=jobs)
        return AnnoyIndex(self._index, self._filedata)

    def add_item(self, vector: list[float], userdata: Any) -> None:
        self._index.add_item(self._i, vector)
        self._filedata.userdata[self._i] = userdata
        self._i += 1


class SentenceChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 120,
        chunk_overlap: int = 30,
        paragraph_tokenizer: Callable[
            [str], list[str]
        ] = tokenize.basic.tokenize_paragraphs,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
    ) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._paragraph_tokenizer = paragraph_tokenizer
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer

    def chunk(self, *, text: str) -> list[str]:
        chunks = []

        buf_words: list[str] = []
        for paragraph in self._paragraph_tokenizer(text):
            last_buf_words: list[str] = []

            for sentence in self._sentence_tokenizer.tokenize(text=paragraph):
                for word in self._word_tokenizer.tokenize(text=sentence):
                    reconstructed = self._word_tokenizer.format_words(
                        buf_words + [word]
                    )

                    if len(reconstructed) > self._max_chunk_size:
                        while (
                            len(self._word_tokenizer.format_words(last_buf_words))
                            > self._chunk_overlap
                        ):
                            last_buf_words = last_buf_words[1:]

                        new_chunk = self._word_tokenizer.format_words(
                            last_buf_words + buf_words
                        )
                        chunks.append(new_chunk)
                        last_buf_words = buf_words
                        buf_words = []

                    buf_words.append(word)

            if buf_words:
                while (
                    len(self._word_tokenizer.format_words(last_buf_words))
                    > self._chunk_overlap
                ):
                    last_buf_words = last_buf_words[1:]

                new_chunk = self._word_tokenizer.format_words(
                    last_buf_words + buf_words
                )
                chunks.append(new_chunk)
                buf_words = []

        return chunks


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
                new_path = f"{path}.{k}" if path else k
                paragraphs.extend(self._extract_paragraphs_from_json(v, new_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                paragraphs.extend(self._extract_paragraphs_from_json(item, new_path))
        elif isinstance(obj, str):
            if len(obj.strip()) > 0:
                paragraphs.append(obj.strip())
        return paragraphs

    async def build_from_json_file(
        self, file_path: Union[str, Path], show_progress: bool = True
    ) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Extract paragraphs from JSON
        raw_paragraphs = self._extract_paragraphs_from_json(json_data)

        # Optional: clean and chunk text
        chunker = SentenceChunker()
        all_chunks = []
        for p in raw_paragraphs:
            chunks = chunker.chunk(text=self._clean_content(p))
            all_chunks.extend(chunks)

        await self.build_from_texts(all_chunks, show_progress)
    def _clean_content(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []

        skip_patterns = [
            'Docs', 'Search', 'GitHub', 'Slack', 'Sign in',
            'Home', 'AI Agents', 'Telephony', 'Recipes', 'Reference',
            'On this page', 'Get started with LiveKit today',
            'Content from https://docs.livekit.io/'
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(pattern in line for pattern in skip_patterns):
                continue
            if line.startswith('http') or line.startswith('[') or line.endswith(']'):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    async def _create_embeddings(
        self, text: str, http_session: Optional[aiohttp.ClientSession] = None
    ) -> openai.EmbeddingData:
        results = await openai.create_embeddings(
            input=[text],
            model=self._embeddings_model,
            dimensions=self._embeddings_dimension,
            http_session=http_session,
        )
        return results[0]

    async def build_from_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as http_session:
            idx_builder = IndexBuilder(
                f=self._embeddings_dimension, metric=self._metric
            )

            cleaned_texts = []
            for text in texts:
                cleaned = self._clean_content(text)
                if cleaned:
                    cleaned_texts.append(cleaned)

            paragraphs_by_uuid = {str(uuid.uuid4()): text for text in cleaned_texts}

            items = paragraphs_by_uuid.items()
            if show_progress:
                items = tqdm(items, desc="Creating embeddings")

            for p_uuid, paragraph in items:
                resp = await self._create_embeddings(paragraph, http_session)
                idx_builder.add_item(resp.embedding, p_uuid)

            logger.info(f"Building index at {self._index_path}")
            idx_builder.build()
            idx_builder.save(str(self._index_path))

            logger.info(f"Saving paragraph data to {self._data_path}")
            with open(self._data_path, "wb") as f:
                pickle.dump(paragraphs_by_uuid, f)

    async def build_from_file(
        self, file_path: Union[str, Path], show_progress: bool = True
    ) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = f.read()

        paragraphs = tokenize.basic.tokenize_paragraphs(raw_data)
        await self.build_from_texts(paragraphs, show_progress)

    @classmethod
    async def create_from_file(
        cls,
        file_path: Union[str, Path],
        index_path: Union[str, Path],
        data_path: Union[str, Path],
        **kwargs,
    ) -> "RAGBuilder":
        builder = cls(index_path=index_path, data_path=data_path, **kwargs)
        await builder.build_from_file(file_path)
        return builder


async def main() -> None:
    """
    Build the RAG database from the structured JSON file.

    Usage:
        1. Run generate_fictional_store_data.py
        2. Run this script to build the RAG database
        3. The database will be created in the 'data' directory
    """
    json_data_path = Path(__file__).parent / "orion_store.json"
    if not json_data_path.exists():
        logger.error(
            "store_data.json not found. Please run generate_fictional_store_data.py first:\n"
            "$ python generate_fictional_store_data.py"
        )
        return

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    logger.info("Building RAG database from JSON...")
    builder = RAGBuilder(
        index_path=output_dir,
        data_path=output_dir / "paragraphs.pkl",
        embeddings_dimension=1536,
    )
    await builder.build_from_json_file(
        file_path=json_data_path,
        show_progress=True,
    )
    logger.info("RAG database successfully built!")
    logger.info(f"Index saved to: {output_dir}")
    logger.info(f"Data saved to: {output_dir / 'paragraphs.pkl'}")

if __name__ == "__main__":
    asyncio.run(main())