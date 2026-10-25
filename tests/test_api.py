from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLLM
from docs_assistant.api import create_app
from docs_assistant.errors import ProviderError
from docs_assistant.services import Services


@pytest.fixture
def client(services: Services) -> TestClient:
    return TestClient(create_app(services.settings, services=services))


def test_index_serves_the_chat_ui_with_security_headers(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>PawPantry Docs Assistant</title>" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_openapi_docs_are_not_broken_by_the_csp(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert client.get("/openapi.json").json()["paths"].keys() >= {"/ask", "/ingest"}


def test_static_assets_are_served(client: TestClient) -> None:
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_health_reports_index_size(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert (body["documents"], body["chunks"]) == (3, 6)
    assert body["llm_model"] == "fake-llm"


def test_documents_lists_the_knowledge_base(client: TestClient) -> None:
    paths = {d["path"] for d in client.get("/documents").json()}
    assert paths == {"shipping.md", "returns.md", "fountain-care.txt"}


def test_ask_returns_answer_citations_and_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "How long is delivery to Germany?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Here is what the docs say [1]."
    assert body["refused"] is False
    assert body["citations"][0]["doc_path"] == "shipping.md"
    assert body["citations"][0]["heading"] == "Delivery times"
    assert body["sources"][0]["cited"] is True
    assert all(not s["cited"] for s in body["sources"][1:])
    assert body["usage"]["input_tokens"] == 120
    assert body["latency_ms"]["total"] >= 0


def test_ask_refusal_shape(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "quantum chromodynamics gluons"}).json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "no_relevant_context"
    assert body["citations"] == [] and body["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "ok", "top_k": 0},
    ],
)
def test_ask_validates_input(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_provider_failure_becomes_502_without_details(services: Services) -> None:
    def explode(system: str, user: str) -> str:
        raise ProviderError("OpenAI", RuntimeError("secret detail"))

    services.assistant.llm = FakeLLM(responder=explode)
    client = TestClient(create_app(services.settings, services=services))
    response = client.post("/ask", json={"question": "How long is delivery to Germany?"})
    assert response.status_code == 502
    assert "secret detail" not in response.text


def test_ingest_without_admin_token_configured(client: TestClient) -> None:
    response = client.post("/ingest", json={})
    assert response.status_code == 200
    assert response.json()["changed"] is False
    assert len(response.json()["unchanged"]) == 3


def test_ingest_requires_the_admin_token_when_configured(services: Services) -> None:
    settings = replace(services.settings, admin_token="s3cret-token")
    client = TestClient(create_app(settings, services=services))
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"X-Admin-Token": "wrong"}).status_code == 401
    ok = client.post("/ingest", json={"rebuild": True}, headers={"X-Admin-Token": "s3cret-token"})
    assert ok.status_code == 200
    assert len(ok.json()["updated"]) == 3
