"""HTTP API and the single-page chat UI."""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import Settings
from .errors import ProviderError
from .services import Services, build_services

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "font-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    ),
}


# --- request / response schemas ---------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["How long do returns take?"])
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str
    snippet: str
    similarity: float


class SourceOut(BaseModel):
    number: int
    chunk_id: str
    doc_path: str
    doc_title: str
    heading: str
    text: str
    similarity: float
    score: float
    cited: bool


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    cost_usd: float | None


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    sources: list[SourceOut]
    model: str | None
    latency_ms: dict[str, float]
    usage: UsageOut


class IngestRequest(BaseModel):
    rebuild: bool = False


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    removed: list[str]
    skipped: list[str]
    chunks_written: int
    embeddings_computed: int
    embeddings_from_cache: int
    embedding_tokens: int
    changed: bool


class DocumentOut(BaseModel):
    path: str
    title: str
    ingested_at: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    llm_model: str
    embedding_model: str


# --- application ------------------------------------------------------------------------


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    svc = services or build_services(settings)
    ingest_lock = threading.Lock()

    app = FastAPI(
        title=f"{settings.company_name} Docs Assistant",
        version=__version__,
        description="Retrieval-augmented answers from your own documents, with citations.",
    )
    app.state.services = svc

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            # FastAPI's interactive docs load Swagger UI / ReDoc from a CDN with inline
            # scripts; the strict CSP applies to everything else.
            if name == "Content-Security-Policy" and request.url.path in _DOCS_PATHS:
                continue
            response.headers.setdefault(name, value)
        return response

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        if not settings.admin_token:
            return
        if x_admin_token is None or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        stats = svc.store.stats()
        return HealthResponse(
            status="ok",
            version=__version__,
            documents=stats["documents"],
            chunks=stats["chunks"],
            llm_model=svc.assistant.llm.model,
            embedding_model=svc.embedder.model,
        )

    @app.get("/documents", response_model=list[DocumentOut])
    def documents() -> list[DocumentOut]:
        return [DocumentOut(**row) for row in svc.store.list_documents()]

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        try:
            result = svc.assistant.ask(body.question, top_k=body.top_k)
        except ProviderError as exc:
            logger.error("Provider failure during /ask: %s", exc)
            raise HTTPException(
                status_code=502, detail="The AI provider is unavailable. Please retry shortly."
            ) from exc
        svc.question_log.record(result)
        return AskResponse.model_validate(result.to_dict())

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    def ingest(body: IngestRequest | None = None) -> IngestResponse:
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An ingestion is already running")
        try:
            report = svc.ingestor.run(rebuild=bool(body and body.rebuild))
        except ProviderError as exc:
            logger.error("Provider failure during /ingest: %s", exc)
            raise HTTPException(status_code=502, detail="Embedding provider failed") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            ingest_lock.release()
        return IngestResponse.model_validate(report.to_dict())

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
