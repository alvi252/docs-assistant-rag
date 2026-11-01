from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLLM
from docs_assistant import cli
from docs_assistant.api import create_app
from docs_assistant.config import Settings
from docs_assistant.questionlog import QuestionLog, read_entries
from docs_assistant.services import Services

DECLINE = json.dumps({"answerable": False, "answer": "The documents do not say."})


@pytest.fixture
def log_path(tmp_path: Path, services: Services) -> Path:
    path = tmp_path / "logs" / "questions.jsonl"
    services.question_log = QuestionLog(path)
    return path


def ask(services: Services, question: str) -> dict[str, object]:
    client = TestClient(create_app(services.settings, services=services))
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200
    return response.json()  # type: ignore[no-any-return]


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_settings_read_the_path_and_empty_means_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUESTION_LOG_PATH", "")
    assert Settings.from_env().question_log_path is None
    monkeypatch.setenv("QUESTION_LOG_PATH", "/data/questions.jsonl")
    assert Settings.from_env().question_log_path == Path("/data/questions.jsonl")


def test_off_by_default_writes_nothing(services: Services, tmp_path: Path) -> None:
    assert services.question_log.path is None
    ask(services, "How long is delivery to Germany?")
    assert not list(tmp_path.glob("**/*.jsonl"))


def test_answered_question_is_logged_with_cited_documents(
    services: Services, log_path: Path
) -> None:
    ask(services, "How long is delivery to Germany?")
    (entry,) = lines(log_path)
    assert set(entry) == {"ts", "question", "refused", "refusal_reason", "cited_docs", "latency_ms"}
    assert entry["question"] == "How long is delivery to Germany?"
    assert entry["refused"] is False and entry["refusal_reason"] is None
    assert entry["cited_docs"] == ["shipping.md"]
    assert isinstance(entry["latency_ms"], float)


def test_gate_refusal_and_model_refusal_are_logged_with_the_reason(
    services: Services, log_path: Path
) -> None:
    ask(services, "quantum chromodynamics gluons")
    services.assistant.llm = FakeLLM(responder=lambda system, user: DECLINE)
    ask(services, "How long is delivery to Germany?")
    gate, model = lines(log_path)
    assert (gate["refused"], gate["refusal_reason"], gate["cited_docs"]) == (
        True,
        "no_relevant_context",
        [],
    )
    assert (model["refused"], model["refusal_reason"]) == (True, "model_declined")


def test_entries_hold_no_answer_text_and_no_request_metadata(
    services: Services, log_path: Path
) -> None:
    client = TestClient(create_app(services.settings, services=services))
    client.post(
        "/ask",
        json={"question": "How long is delivery to Germany?"},
        headers={"Authorization": "Basic c2VjcmV0", "X-Forwarded-For": "203.0.113.9"},
    )
    raw = log_path.read_text(encoding="utf-8")
    assert "Here is what the docs say" not in raw
    assert "203.0.113.9" not in raw and "c2VjcmV0" not in raw


def test_invalid_requests_are_not_logged(services: Services, log_path: Path) -> None:
    client = TestClient(create_app(services.settings, services=services))
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert not log_path.exists()


def test_unwritable_log_does_not_break_the_answer(services: Services, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    services.question_log = QuestionLog(blocker / "questions.jsonl")
    assert ask(services, "How long is delivery to Germany?")["refused"] is False


def test_read_entries_skips_damaged_lines_and_filters_refused(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"question": "a", "refused": false}\n\nnot json\n[1]\n'
        '{"question": "b", "refused": true}\n',
        encoding="utf-8",
    )
    assert [e["question"] for e in read_entries(path)] == ["a", "b"]
    assert [e["question"] for e in read_entries(path, refused_only=True)] == ["b"]


def test_cli_lists_questions_and_filters_refused(
    services: Services,
    log_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ask(services, "How long is delivery to Germany?")
    ask(services, "quantum chromodynamics gluons")
    monkeypatch.setenv("QUESTION_LOG_PATH", str(log_path))
    monkeypatch.chdir(log_path.parent)  # no .env here

    assert cli.main(["questions"]) == 0
    everything = capsys.readouterr()
    assert "answered" in everything.out and "cited: shipping.md" in everything.out
    assert "quantum chromodynamics gluons" in everything.out
    assert "2 question(s)" in everything.err

    assert cli.main(["questions", "--refused"]) == 0
    refused = capsys.readouterr().out
    assert "REFUSED (no_relevant_context)" in refused
    assert "How long is delivery to Germany?" not in refused

    assert cli.main(["questions", "--refused", "--json"]) == 0
    (entry,) = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert entry["question"] == "quantum chromodynamics gluons"


def test_cli_explains_when_the_log_is_off_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUESTION_LOG_PATH", "")
    assert cli.main(["questions"]) == 2
    assert "QUESTION_LOG_PATH" in capsys.readouterr().err
    monkeypatch.setenv("QUESTION_LOG_PATH", str(tmp_path / "none.jsonl"))
    assert cli.main(["questions"]) == 0
    assert "No questions logged yet" in capsys.readouterr().err
