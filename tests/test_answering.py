from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FakeLLM
from docs_assistant.answering import (
    REFUSAL_MESSAGES,
    detect_language,
    extract_citations,
    parse_model_output,
)
from docs_assistant.models import Chunk, RetrievedChunk
from docs_assistant.prompts import ANSWER_SCHEMA
from docs_assistant.services import Services
from docs_assistant.usage import UsageLog


def _source(n: int, doc: str = "doc.md", heading: str = "H") -> RetrievedChunk:
    chunk = Chunk(
        id=f"c{n}",
        doc_path=doc,
        doc_title=doc.title(),
        heading=heading,
        ordinal=n,
        text=f"text {n}",
    )
    return RetrievedChunk(chunk=chunk, similarity=0.5, score=0.1)


# ---- citation mapping -------------------------------------------------------------------


def test_citations_map_to_document_and_heading() -> None:
    sources = [_source(1, "shipping.md", "Delivery"), _source(2, "returns.md", "Window")]
    text, citations = extract_citations("Ships in 3 days [1]. Returns within 30 days [2].", sources)
    assert text == "Ships in 3 days [1]. Returns within 30 days [2]."
    assert [(c.number, c.doc_path, c.heading, c.chunk_id) for c in citations] == [
        (1, "shipping.md", "Delivery", "c1"),
        (2, "returns.md", "Window", "c2"),
    ]
    assert citations[0].snippet == "text 1"


def test_grouped_and_repeated_citations_are_normalised() -> None:
    sources = [_source(1), _source(2), _source(3)]
    text, citations = extract_citations("A [3, 1]. B [1][3]. C [2; 3].", sources)
    assert text == "A [3][1]. B [1][3]. C [2][3]."
    assert [c.number for c in citations] == [3, 1, 2]  # order of first appearance, unique


def test_out_of_range_citations_are_removed() -> None:
    text, citations = extract_citations("Fact [1]. Invented [7] .", [_source(1)])
    assert text == "Fact [1]. Invented."
    assert [c.number for c in citations] == [1]


# ---- model output parsing ----------------------------------------------------------------


def test_parse_json_reply() -> None:
    out = parse_model_output('{"answerable": false, "answer": "Not in the docs."}')
    assert (out.answerable, out.answer) == (False, "Not in the docs.")


def test_parse_json_wrapped_in_prose() -> None:
    out = parse_model_output('Sure!\n```json\n{"answerable": true, "answer": "Yes [1]."}\n```')
    assert (out.answerable, out.answer) == (True, "Yes [1].")


def test_parse_plain_text_fallback() -> None:
    assert parse_model_output("It ships in 2 days [1].").answerable is True
    assert parse_model_output("I don't know based on the documents.").answerable is False


# ---- the pipeline -------------------------------------------------------------------------


def test_answer_with_citations(services: Services, fake_llm: FakeLLM) -> None:
    result = services.assistant.ask("How long does delivery to Germany take?")

    assert not result.refused and result.answerable
    assert result.answer == "Here is what the docs say [1]."
    assert len(result.citations) == 1
    assert result.citations[0].doc_path == result.sources[0].chunk.doc_path == "shipping.md"
    assert result.citations[0].heading == "Delivery times"
    assert set(result.latency_ms) == {"retrieval", "generation", "total"}
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 30
    assert result.usage.cost_usd is None  # fake models have no price

    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call["schema"] is ANSWER_SCHEMA
    assert "[1] Source: Shipping › Delivery times" in str(call["user"])
    assert "Question: How long does delivery to Germany take?" in str(call["user"])
    assert "ONLY the numbered context passages" in str(call["system"])


def test_no_relevant_context_refuses_without_calling_the_model(
    services: Services, fake_llm: FakeLLM
) -> None:
    result = services.assistant.ask("Explain quantum chromodynamics lattice gluons")
    assert result.refused
    assert result.refusal_reason == "no_relevant_context"
    assert result.answer == REFUSAL_MESSAGES["en"]
    assert result.citations == [] and result.sources == []
    assert result.model is None
    assert fake_llm.calls == []


def test_gate_refusal_follows_the_question_language(services: Services) -> None:
    result = services.assistant.ask("Quelle est la recette de la quiche lorraine ?")
    assert result.refusal_reason == "no_relevant_context"
    assert result.answer == REFUSAL_MESSAGES["fr"]


def test_model_declining_is_a_refusal_but_keeps_sources(
    services: Services, fake_llm: FakeLLM
) -> None:
    fake_llm.responder = lambda system, user: json.dumps(
        {"answerable": False, "answer": "I don't know based on the documents [1]."}
    )
    result = services.assistant.ask("Does standard delivery to Germany include parcel insurance?")
    assert result.refused
    assert result.refusal_reason == "model_declined"
    assert result.citations == []
    assert result.answer == "I don't know based on the documents."  # marker stripped
    assert result.sources  # the passages that were checked are still shown
    assert len(fake_llm.calls) == 1


def test_empty_question_is_rejected(services: Services) -> None:
    with pytest.raises(ValueError):
        services.assistant.ask("   ")


def test_usage_log_records_counts_but_no_text(services: Services, tmp_path: Path) -> None:
    log_path = tmp_path / "usage.jsonl"
    services.assistant.usage_log = UsageLog(log_path)
    services.assistant.ask("How long does delivery to Germany take?")
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert entries == [
        {
            "ts": entries[0]["ts"],
            "kind": "chat",
            "model": "fake-llm",
            "input_tokens": 120,
            "output_tokens": 30,
            "cost_usd": None,
        }
    ]
    assert "Germany" not in log_path.read_text()


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Wie lange dauert die Lieferung nach Österreich und was kostet sie?", "de"),
        ("¿Cuánto tarda el envío a España?", "es"),
        ("Hoe lang duurt de levering en wat kost het?", "nl"),
        ("12345", "en"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang
