"""Provider adapters, exercised with stub SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from docs_assistant.embeddings import HashingEmbeddings, OpenAIEmbeddings
from docs_assistant.errors import ProviderError
from docs_assistant.llm import AnthropicChat, OpenAIChat
from docs_assistant.prompts import ANSWER_SCHEMA
from docs_assistant.usage import estimate_cost


class _Recorder:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        if self.error:
            raise self.error
        return self.response(kwargs) if callable(self.response) else self.response


def test_openai_chat_request_and_response_mapping() -> None:
    completions = _Recorder(
        SimpleNamespace(
            model="gpt-4.1-mini-2025-04-14",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "x"}'))],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=60),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    chat = OpenAIChat("gpt-4.1-mini", client=client)

    reply = chat.complete(
        system="S", user="U", max_tokens=300, temperature=0.0, schema=ANSWER_SCHEMA
    )

    sent = completions.kwargs[0]
    assert sent["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "strict": True, "schema": ANSWER_SCHEMA},
    }
    assert sent["max_completion_tokens"] == 300
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"answer": "x"}', 900, 60)
    assert reply.model == "gpt-4.1-mini-2025-04-14"


class _FakeAuthError(Exception):
    status_code = 401


def test_provider_errors_are_wrapped_without_echoing_details() -> None:
    secret_ish = "Incorrect API key provided: fake-credential-should-not-leak"
    completions = _Recorder(error=_FakeAuthError(secret_ish))
    chat = OpenAIChat(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with pytest.raises(ProviderError) as info:
        chat.complete(system="S", user="U", max_tokens=10, temperature=0)
    assert "fake-credential" not in str(info.value)
    assert "HTTP 401" in str(info.value)
    assert info.value.status_code == 401


def test_anthropic_chat_joins_text_blocks() -> None:
    messages = _Recorder(
        SimpleNamespace(
            model="claude-test",
            content=[
                SimpleNamespace(type="text", text='{"answerable": true, '),
                SimpleNamespace(type="thinking", text="ignored"),
                SimpleNamespace(type="text", text='"answer": "ok [1]"}'),
            ],
            usage=SimpleNamespace(input_tokens=500, output_tokens=20),
        )
    )
    chat = AnthropicChat("claude-test", client=SimpleNamespace(messages=messages))

    reply = chat.complete(
        system="S", user="U", max_tokens=200, temperature=0.0, schema=ANSWER_SCHEMA
    )

    sent = messages.kwargs[0]
    assert sent["system"] == "S"
    assert sent["messages"] == [{"role": "user", "content": "U"}]
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    assert "temperature" not in sent
    assert reply.text == '{"answerable": true, "answer": "ok [1]"}'
    assert (reply.input_tokens, reply.output_tokens) == (500, 20)


def test_openai_embeddings_batch_and_restore_order() -> None:
    def respond(kwargs: dict[str, Any]) -> Any:
        batch = kwargs["input"]
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
            for i, text in enumerate(batch)
        ]
        return SimpleNamespace(
            data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=len(batch))
        )

    embeddings_api = _Recorder(respond)
    provider = OpenAIEmbeddings(
        "text-embedding-3-small", batch_size=2, client=SimpleNamespace(embeddings=embeddings_api)
    )
    result = provider.embed(["a", "bbb", "cc"])

    assert [len(k["input"]) for k in embeddings_api.kwargs] == [2, 1]
    assert result.tokens == 3
    assert result.vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    expected_first = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(result.vectors[0], expected_first)  # "a" -> [1, 1], order restored


def test_hashing_embeddings_are_deterministic_and_lexical() -> None:
    provider = HashingEmbeddings(dim=256)
    a, b, c = provider.embed(["fountain filter", "fountain filter", "parcel delivery"]).vectors
    assert np.allclose(a, b)
    assert float(a @ c) < 0.5 < float(a @ b)


@pytest.mark.parametrize(
    ("model", "tokens_in", "tokens_out", "expected"),
    [
        ("gpt-4.1-mini", 1000, 200, 0.00072),
        ("text-embedding-3-small", 50_000, 0, 0.001),
        ("unknown-model", 1, 1, None),
    ],
)
def test_cost_estimates(
    model: str, tokens_in: int, tokens_out: int, expected: float | None
) -> None:
    cost = estimate_cost(model, tokens_in, tokens_out)
    assert cost == (pytest.approx(expected) if expected is not None else None)
