# rag_index.py
import pickle
from typing import Any, Iterable, Literal
from dataclasses import dataclass
from pathlib import Path
import annoy
from livekit.agents import tokenize

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
        with open(p / METADATA_FILE, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(p / ANNOY_FILE))
        return cls(index, metadata)

    def size(self) -> int:
        return self._index.get_n_items()

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            yield Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i),
            )

    def query(self, vector: list[float], n: int, search_k: int = -1) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(vector, n, search_k=search_k, include_distances=True)
        return [QueryResult(userdata=self._filedata.userdata[i], distance=distance) for i, distance in zip(*ids)]

class IndexBuilder:
    def __init__(self, f: int, metric: Metric) -> None:
        self._index = annoy.AnnoyIndex(f, metric)
        self._filedata = _FileData(f=f, metric=metric, userdata={})
        self._i = 0

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._index.save(str(p / ANNOY_FILE))
        with open(p / METADATA_FILE, "wb") as f:
            pickle.dump(self._filedata, f)

    def build(self, trees: int = 50, jobs: int = -1) -> AnnoyIndex:
        self._index.build(n_trees=trees, n_jobs=jobs)
        return AnnoyIndex(self._index, self._filedata)

    def add_item(self, vector: list[float], userdata: Any) -> None:
        self._index.add_item(self._i, vector)
        self._filedata.userdata[self._i] = userdata
        self._i += 1

class SentenceChunker:
    def __init__(self, *, max_chunk_size: int = 120, chunk_overlap: int = 30):
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._paragraph_tokenizer = tokenize.basic.tokenize_paragraphs
        self._sentence_tokenizer = tokenize.basic.SentenceTokenizer()
        self._word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

    def chunk(self, *, text: str) -> list[str]:
        chunks = []
        buf_words: list[str] = []

        for paragraph in self._paragraph_tokenizer(text):
            last_buf_words: list[str] = []

            for sentence in self._sentence_tokenizer.tokenize(text=paragraph):
                for word in self._word_tokenizer.tokenize(text=sentence):
                    reconstructed = self._word_tokenizer.format_words(buf_words + [word])
                    if len(reconstructed) > self._max_chunk_size:
                        while len(self._word_tokenizer.format_words(last_buf_words)) > self._chunk_overlap:
                            last_buf_words = last_buf_words[1:]
                        chunks.append(self._word_tokenizer.format_words(last_buf_words + buf_words))
                        last_buf_words = buf_words
                        buf_words = []
                    buf_words.append(word)

            if buf_words:
                while len(self._word_tokenizer.format_words(last_buf_words)) > self._chunk_overlap:
                    last_buf_words = last_buf_words[1:]
                chunks.append(self._word_tokenizer.format_words(last_buf_words + buf_words))
                buf_words = []

        return chunks
