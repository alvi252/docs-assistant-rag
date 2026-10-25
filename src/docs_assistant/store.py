"""SQLite persistence for documents, chunks, vectors and the embedding cache.

A single file, no server. Vectors are float32 blobs; search happens in NumPy (see
``retrieval.py``). That is plenty for thousands of chunks; for millions, swap this class for
pgvector, Qdrant or similar behind the same methods.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    heading     TEXT NOT NULL,
    text        TEXT NOT NULL,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.shape[0] != dim:
        raise ValueError(f"Corrupt vector: expected {dim} dims, got {vector.shape[0]}")
    return vector


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection per operation: safe across FastAPI worker threads."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:  # commits on success, rolls back on exception
                yield conn
        finally:
            conn.close()

    # --- documents -------------------------------------------------------------------

    def fingerprints(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT path, fingerprint FROM documents").fetchall()
        return {row["path"]: row["fingerprint"] for row in rows}

    def replace_document(
        self,
        *,
        path: str,
        title: str,
        fingerprint: str,
        model: str,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Atomically swap a document's chunks: readers see the old or the new version."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE path = ?", (path,))
            cursor = conn.execute(
                "INSERT INTO documents (path, title, fingerprint, ingested_at) VALUES (?, ?, ?, ?)",
                (path, title, fingerprint, now),
            )
            document_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO chunks"
                " (id, document_id, ordinal, heading, text, model, dim, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, document_id, c.ordinal, c.heading, c.text, model, len(v), _to_blob(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_documents(self, paths: Iterable[str]) -> None:
        with self.connect() as conn:
            conn.executemany("DELETE FROM documents WHERE path = ?", [(p,) for p in paths])

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT d.path, d.title, d.ingested_at, COUNT(c.id) AS chunks"
                " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
                " GROUP BY d.id ORDER BY d.title"
            ).fetchall()
        return [dict(row) for row in rows]

    def load_index(self) -> tuple[list[Chunk], np.ndarray, str | None]:
        """Return every chunk, their vectors as one matrix, and the embedding model name."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.ordinal, c.heading, c.text, c.model, c.dim, c.embedding,"
                " d.path, d.title FROM chunks c JOIN documents d ON d.id = c.document_id"
                " ORDER BY d.path, c.ordinal"
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32), None
        models = {row["model"] for row in rows}
        if len(models) > 1:
            raise RuntimeError(f"Index mixes embedding models {sorted(models)}; re-ingest it.")
        chunks = [
            Chunk(
                id=row["id"],
                doc_path=row["path"],
                doc_title=row["title"],
                heading=row["heading"],
                ordinal=row["ordinal"],
                text=row["text"],
            )
            for row in rows
        ]
        matrix = np.vstack([_from_blob(row["embedding"], row["dim"]) for row in rows])
        return chunks, matrix, models.pop()

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(documents), "chunks": int(chunks)}

    # --- embedding cache ---------------------------------------------------------------

    def cached_embeddings(self, model: str, text_hashes: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self.connect() as conn:
            for start in range(0, len(text_hashes), 500):
                batch = list(text_hashes[start : start + 500])
                placeholders = ",".join("?" * len(batch))
                # Only "?" placeholders are interpolated; values are bound parameters.
                query = (
                    "SELECT text_hash, dim, vector FROM embedding_cache"  # noqa: S608
                    f" WHERE model = ? AND text_hash IN ({placeholders})"
                )
                rows = conn.execute(query, [model, *batch]).fetchall()
                for row in rows:
                    found[row["text_hash"]] = _from_blob(row["vector"], row["dim"])
        return found

    def cache_embeddings(self, model: str, items: dict[str, np.ndarray]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, dim, vector)"
                " VALUES (?, ?, ?, ?)",
                [(model, h, len(v), _to_blob(v)) for h, v in items.items()],
            )

    # --- change tracking ---------------------------------------------------------------

    def generation(self) -> int:
        """A counter bumped after every change, so readers know when to reload."""
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
        return int(row["value"]) if row else 0

    def bump_generation(self) -> int:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('generation', '1')"
                " ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
            )
            row = conn.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
        return int(row["value"])
