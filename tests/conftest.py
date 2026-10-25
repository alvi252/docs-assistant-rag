"""Shared fixtures: a tiny knowledge base and fake providers. No test touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from docs_assistant.config import Settings
from docs_assistant.embeddings import EmbeddingResult, HashingEmbeddings
from docs_assistant.llm import LLMResponse
from docs_assistant.services import Services, build_services

SHIPPING_MD = """\
---
title: Shipping
---

# Shipping

We ship parcels from our warehouse to customers across Europe.

## Delivery times

Standard delivery to Germany takes 3 to 5 business days and costs 6.90 euros.
Delivery to France takes 2 to 3 business days.

## Tracking

Every parcel gets a tracking link by email when it leaves the warehouse.
"""

RETURNS_MD = """\
# Returns and refunds

## Return window

Unopened items can be returned within 30 days of delivery for a refund.

## Refund timing

Refunds are issued within 5 business days after the returned parcel is inspected.
"""

CARE_TXT = """\
Rinse the fountain filter under cold water. Replace the fountain filter every four weeks.
"""


def write_docs(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "shipping.md").write_text(SHIPPING_MD, encoding="utf-8")
    (directory / "returns.md").write_text(RETURNS_MD, encoding="utf-8")
    (directory / "fountain-care.txt").write_text(CARE_TXT, encoding="utf-8")
    return directory


class CountingEmbeddings:
    """Hashing embeddings that record every batch sent to the 'provider'."""

    def __init__(self) -> None:
        self.inner = HashingEmbeddings()
        self.calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return self.inner.model

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        return self.inner.embed(texts)

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)


@dataclass
class FakeLLM:
    """Scripted chat model. By default it answers from passage [1] and cites it."""

    responder: Callable[[str, str], str] | None = None
    model_name: str = "fake-llm"
    calls: list[dict[str, object]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.model_name

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.responder is not None:
            text = self.responder(system, user)
        else:
            text = json.dumps({"answerable": True, "answer": "Here is what the docs say [1]."})
        return LLMResponse(text=text, model=self.model_name, input_tokens=120, output_tokens=30)


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    return write_docs(tmp_path / "docs")


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path) -> Settings:
    return Settings(
        docs_dir=docs_dir,
        index_path=tmp_path / "index.sqlite",
        embedding_provider="hashing",
        min_similarity=0.2,
        top_k=3,
    )


@pytest.fixture
def embeddings() -> CountingEmbeddings:
    return CountingEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings: Settings, embeddings: CountingEmbeddings, fake_llm: FakeLLM) -> Services:
    svc = build_services(settings, embedding_provider=embeddings, llm=fake_llm)
    svc.ingestor.run()
    return svc


def with_settings(settings: Settings, **changes: object) -> Settings:
    return replace(settings, **changes)  # type: ignore[arg-type]
