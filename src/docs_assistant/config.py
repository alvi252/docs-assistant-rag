"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw and raw.strip() else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    """All tunables in one place. Every field maps to an environment variable."""

    company_name: str = "PawPantry"
    docs_dir: Path = Path("data/docs")
    index_path: Path = Path("data/index.sqlite")

    # Providers
    llm_provider: str = "openai"  # "openai" | "anthropic"
    llm_model: str = "gpt-4.1-mini"
    embedding_provider: str = "openai"  # "openai" | "hashing" (offline, for tests/demos)
    embedding_model: str = "text-embedding-3-small"
    request_timeout_s: float = 30.0

    # Chunking
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    # Retrieval
    top_k: int = 5
    min_similarity: float = 0.30
    hybrid: bool = True

    # Generation
    max_answer_tokens: int = 500
    temperature: float = 0.0

    # Operations
    admin_token: str = field(default="", repr=False)
    usage_log_path: Path | None = None
    question_log_path: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        usage_log = os.environ.get("USAGE_LOG_PATH", "").strip()
        question_log = os.environ.get("QUESTION_LOG_PATH", "").strip()
        provider = _env_str("LLM_PROVIDER", cls.llm_provider).lower()
        default_model = cls.llm_model if provider == "openai" else "claude-haiku-5-5"
        return cls(
            company_name=_env_str("COMPANY_NAME", cls.company_name),
            docs_dir=Path(_env_str("DOCS_DIR", str(cls.docs_dir))),
            index_path=Path(_env_str("INDEX_PATH", str(cls.index_path))),
            llm_provider=provider,
            llm_model=_env_str("LLM_MODEL", default_model),
            embedding_provider=_env_str("EMBEDDING_PROVIDER", cls.embedding_provider).lower(),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", cls.request_timeout_s),
            chunk_max_chars=_env_int("CHUNK_MAX_CHARS", cls.chunk_max_chars),
            chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", cls.chunk_overlap_chars),
            top_k=_env_int("TOP_K", cls.top_k),
            min_similarity=_env_float("MIN_SIMILARITY", cls.min_similarity),
            hybrid=_env_bool("HYBRID_SEARCH", cls.hybrid),
            max_answer_tokens=_env_int("MAX_ANSWER_TOKENS", cls.max_answer_tokens),
            temperature=_env_float("TEMPERATURE", cls.temperature),
            admin_token=os.environ.get("ADMIN_TOKEN", "").strip(),
            usage_log_path=Path(usage_log) if usage_log else None,
            question_log_path=Path(question_log) if question_log else None,
        )


def load_dotenv_if_present(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env without overriding variables already set."""
    if not path.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
