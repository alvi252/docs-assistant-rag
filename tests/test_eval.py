from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FakeLLM
from docs_assistant.eval import EvalReport, load_questions, render_html, render_markdown, run_eval
from docs_assistant.services import Services

QUESTIONS = """\
questions:
  - id: a1
    question: How long does standard delivery to Germany take?
    expected_sources: [shipping.md]
    expected_facts: [["3 to 5", "3-5"], ["6.90"]]
  - id: a2
    question: Within how many days can unopened items be returned?
    expected_sources: [returns.md]
    expected_facts: [["30 days"]]
  - id: u1
    question: Explain quantum chromodynamics lattice gluons
    answerable: false
  - id: u2
    question: Does standard delivery to Germany include parcel insurance?
    answerable: false
"""


def _responder(system: str, user: str) -> str:
    question = user.rsplit("Question:", 1)[1]
    if "insurance" in question:
        return json.dumps({"answerable": False, "answer": "I don't know based on the documents."})
    if "Germany" in question:
        return json.dumps({"answerable": True, "answer": "It takes 3 to 5 business days [1]."})
    return json.dumps({"answerable": True, "answer": "Within 30 days [1]."})


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(QUESTIONS)
    return path


def test_eval_metrics_with_fake_models(services: Services, questions_file: Path) -> None:
    services.assistant.llm = FakeLLM(responder=_responder)
    report = run_eval(services.assistant, load_questions(questions_file))
    summary = report.summary()

    assert summary["hits"] == 2 and summary["hit_rate"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["refusal_correct"] == 4
    assert summary["refused_correctly"] == 2
    assert summary["facts_found"] == 2 and summary["facts_total"] == 3  # price was not mentioned
    assert summary["cites_expected"] == 2

    by_id = {o.id: o for o in report.outcomes}
    assert by_id["u1"].refusal_reason == "no_relevant_context"
    assert by_id["u2"].refusal_reason == "model_declined"

    markdown = render_markdown(report)
    assert "| Retrieval hit-rate@3 (answerable questions) | 2/2 (100%) |" in markdown
    assert "| u2 |" in markdown
    html = render_html(report)
    assert "Evaluation report" in html and "Hit-rate@3" in html


def test_retrieval_only_mode_makes_no_chat_calls(services: Services, questions_file: Path) -> None:
    llm = FakeLLM(responder=_responder)
    services.assistant.llm = llm
    report = run_eval(services.assistant, load_questions(questions_file), retrieval_only=True)
    assert llm.calls == []
    assert report.mode == "retrieval-only"
    assert report.summary()["hits"] == 2
    assert "Expected facts" not in render_markdown(report)


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("- {id: x, question: a}\n- {id: x, question: b}\n")
    with pytest.raises(ValueError, match="unique"):
        load_questions(path)


def test_report_round_trips_through_json(services: Services, questions_file: Path) -> None:
    services.assistant.llm = FakeLLM(responder=_responder)
    report = run_eval(services.assistant, load_questions(questions_file))
    restored = EvalReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.summary() == report.summary()
    assert render_markdown(restored) == render_markdown(report)
