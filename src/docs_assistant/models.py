"""Plain data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RefusalReason = Literal["no_relevant_context", "model_declined"]


@dataclass(frozen=True)
class Section:
    """A piece of a document under one heading path, before size-based splitting."""

    heading: str
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    path: str  # relative to the docs directory, POSIX separators
    title: str
    sections: list[Section]
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_path: str
    doc_title: str
    heading: str
    ordinal: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: title and heading give the vector context."""
        return f"{self.doc_title} — {self.heading}\n\n{self.text}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # cosine similarity between query and chunk, -1..1
    score: float  # final ranking score (RRF score in hybrid mode, cosine otherwise)
    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] marker used in the answer
    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str
    snippet: str
    similarity: float


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float | None = 0.0

    def add_cost(self, cost: float | None) -> None:
        if cost is None or self.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += cost


@dataclass
class AskResult:
    question: str
    answer: str
    answerable: bool
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    sources: list[RetrievedChunk]  # every chunk that was given to the model, numbered 1..n
    model: str | None
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    @property
    def refused(self) -> bool:
        return self.refusal_reason is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answerable": self.answerable,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations": [asdict(c) for c in self.citations],
            "sources": [
                {
                    "number": i,
                    "chunk_id": s.chunk.id,
                    "doc_path": s.chunk.doc_path,
                    "doc_title": s.chunk.doc_title,
                    "heading": s.chunk.heading,
                    "text": s.chunk.text,
                    "similarity": round(s.similarity, 4),
                    "score": round(s.score, 6),
                    "cited": any(c.number == i for c in self.citations),
                }
                for i, s in enumerate(self.sources, start=1)
            ],
            "model": self.model,
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
            "usage": asdict(self.usage),
        }


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_from_cache: int = 0
    embedding_tokens: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed"] = self.changed
        return data
