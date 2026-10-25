"""Chat-completion providers behind one small interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .errors import ProviderError

JsonSchema = dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        """Return the model's text. With ``schema``, the text is JSON matching it."""
        ...


class OpenAIChat:
    """OpenAI Chat Completions with Structured Outputs (tested live with ``gpt-4.1-mini``)."""

    def __init__(
        self, model: str = "gpt-4.1-mini", *, timeout_s: float = 30.0, client: Any | None = None
    ) -> None:
        self._model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_s, max_retries=2)  # reads OPENAI_API_KEY
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": True, "schema": schema},
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError("OpenAI", exc) from exc
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model or self._model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class AnthropicChat:
    """Anthropic Messages API with structured JSON output.

    Implemented and unit-tested against a stub client, but not exercised live in this demo.
    Install the extra (``pip install '.[anthropic]'``) and set ``ANTHROPIC_API_KEY``.
    ``temperature`` is accepted for interface compatibility but not sent: the current Messages
    API exposes no sampling-temperature parameter.
    """

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        timeout_s: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(timeout=timeout_s, max_retries=2)  # reads ANTHROPIC_API_KEY
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: JsonSchema | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ProviderError("Anthropic", exc) from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = response.usage
        return LLMResponse(
            text=text,
            model=getattr(response, "model", None) or self._model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def build_llm_provider(name: str, model: str, timeout_s: float) -> LLMProvider:
    if name == "openai":
        return OpenAIChat(model, timeout_s=timeout_s)
    if name == "anthropic":
        return AnthropicChat(model, timeout_s=timeout_s)
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r} (expected 'openai' or 'anthropic')")
