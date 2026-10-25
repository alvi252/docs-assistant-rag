"""The question-answering pipeline: retrieve, decide, generate, map citations."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .llm import LLMProvider
from .models import AskResult, Citation, RefusalReason, RetrievedChunk, Usage
from .prompts import ANSWER_SCHEMA, build_system_prompt, build_user_message
from .retrieval import Retriever
from .usage import UsageLog, estimate_cost

logger = logging.getLogger(__name__)

_GROUPED_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)+)\s*\]")
_CITATION = re.compile(r"\[(\d+)\]")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

REFUSAL_MESSAGES = {
    "en": "I don't know based on the available documents. Try rephrasing your question, "
    "or contact our customer care team.",
    "fr": "Je ne sais pas d'après les documents disponibles. Essayez de reformuler votre "
    "question ou contactez notre service client.",
    "de": "Das kann ich anhand der verfügbaren Dokumente nicht beantworten. Bitte formulieren "
    "Sie die Frage um oder wenden Sie sich an unseren Kundenservice.",
    "es": "No lo sé según los documentos disponibles. Intente reformular su pregunta o "
    "contacte con nuestro servicio de atención al cliente.",
    "it": "Non lo so in base ai documenti disponibili. Prova a riformulare la domanda o "
    "contatta il nostro servizio clienti.",
    "nl": "Dat weet ik niet op basis van de beschikbare documenten. Probeer uw vraag anders "
    "te formuleren of neem contact op met onze klantenservice.",
}

_LANGUAGE_HINTS = {
    "en": "the is are what how do does can i my you your to for of and when where which with",
    "fr": "le la les est sont quel quelle quels comment je mon ma mes vous votre pour des et "
    "quand où combien puis-je un une du ai ce avec pas",
    "de": "der die das ist sind wie was ich mein meine sie ihr für und wann wo kann ein eine "
    "nicht zu mit",
    "es": "el los las es son qué cómo yo mi mis usted su para y cuándo dónde puedo un una "
    "cuánto con",
    "it": "il lo gli è sono come cosa io mio mia lei per e quando dove posso un una quanto con",
    "nl": "de het is zijn wat hoe ik mijn u uw voor en wanneer waar kan een niet met",
}
_LANGUAGE_WORDS = {lang: frozenset(words.split()) for lang, words in _LANGUAGE_HINTS.items()}


def detect_language(text: str) -> str:
    """Cheap stopword vote, only used for the canned refusal (the model handles the rest).

    >>> detect_language("Combien de temps ai-je pour retourner un article ?")
    'fr'
    >>> detect_language("How long does delivery take?")
    'en'
    """
    words = re.findall(r"[\w'-]+", text.lower())
    scores = {lang: sum(w in vocab for w in words) for lang, vocab in _LANGUAGE_WORDS.items()}
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] > 0 else "en"


@dataclass(frozen=True)
class ModelOutput:
    answerable: bool
    answer: str


def parse_model_output(text: str) -> ModelOutput:
    """Read the model's JSON reply; tolerate prose around it or plain text."""
    candidates = [text]
    match = _JSON_OBJECT.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            answerable = data.get("answerable", True)
            return ModelOutput(answerable=answerable is not False, answer=data["answer"].strip())
    logger.warning("Model reply was not valid JSON; using it as plain text")
    plain = text.strip()
    declined = plain.lower().startswith(("i don't know", "i do not know"))
    return ModelOutput(answerable=not declined, answer=plain)


def extract_citations(answer: str, sources: Sequence[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Map [n] markers to sources. Invalid numbers are removed from the text.

    Grouped markers such as "[1, 3]" are normalised to "[1][3]" first.
    """
    answer = _GROUPED_CITATION.sub(
        lambda m: "".join(f"[{n.strip()}]" for n in re.split(r"[,;]", m.group(1))), answer
    )
    citations: dict[int, Citation] = {}

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(sources):
            return ""
        if number not in citations:
            source = sources[number - 1]
            citations[number] = Citation(
                number=number,
                chunk_id=source.chunk.id,
                doc_path=source.chunk.doc_path,
                doc_title=source.chunk.doc_title,
                heading=source.chunk.heading,
                snippet=source.chunk.text,
                similarity=round(source.similarity, 4),
            )
        return match.group(0)

    cleaned = _CITATION.sub(replace, answer)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, list(citations.values())


class DocsAssistant:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        company: str = "PawPantry",
        max_answer_tokens: int = 500,
        temperature: float = 0.0,
        usage_log: UsageLog | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = build_system_prompt(company)
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.usage_log = usage_log or UsageLog(None)

    def ask(self, question: str, *, top_k: int | None = None) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        started = time.perf_counter()
        search = self.retriever.search(question, top_k)
        retrieved_at = time.perf_counter()

        usage = Usage(embedding_tokens=search.embedding_tokens)
        usage.add_cost(estimate_cost(self.retriever.embedder.model, search.embedding_tokens))

        if not search.chunks:
            # Nothing passed the relevance gate: refuse without paying for a completion.
            return self._result(
                question,
                answer=REFUSAL_MESSAGES[detect_language(question)],
                reason="no_relevant_context",
                sources=[],
                citations=[],
                model=None,
                usage=usage,
                timings=(started, retrieved_at, retrieved_at),
            )

        reply = self.llm.complete(
            system=self.system_prompt,
            user=build_user_message(question, search.chunks),
            max_tokens=self.max_answer_tokens,
            temperature=self.temperature,
            schema=ANSWER_SCHEMA,
        )
        generated_at = time.perf_counter()
        self.usage_log.record("chat", reply.model, reply.input_tokens, reply.output_tokens)
        usage.input_tokens = reply.input_tokens
        usage.output_tokens = reply.output_tokens
        usage.add_cost(estimate_cost(reply.model, reply.input_tokens, reply.output_tokens))

        output = parse_model_output(reply.text)
        # A refusal must not cite anything: with no valid sources every [n] marker is dropped.
        answer, citations = extract_citations(
            output.answer, search.chunks if output.answerable else []
        )
        if not answer:
            answer = REFUSAL_MESSAGES[detect_language(question)]
        reason: RefusalReason | None = None if output.answerable else "model_declined"
        return self._result(
            question,
            answer=answer,
            reason=reason,
            sources=search.chunks,
            citations=citations,
            model=reply.model,
            usage=usage,
            timings=(started, retrieved_at, generated_at),
        )

    @staticmethod
    def _result(
        question: str,
        *,
        answer: str,
        reason: RefusalReason | None,
        sources: list[RetrievedChunk],
        citations: list[Citation],
        model: str | None,
        usage: Usage,
        timings: tuple[float, float, float],
    ) -> AskResult:
        started, retrieved_at, generated_at = timings
        result = AskResult(
            question=question,
            answer=answer,
            answerable=reason is None,
            refusal_reason=reason,
            citations=citations,
            sources=sources,
            model=model,
            latency_ms={
                "retrieval": (retrieved_at - started) * 1000,
                "generation": (generated_at - retrieved_at) * 1000,
                "total": (time.perf_counter() - started) * 1000,
            },
            usage=usage,
        )
        logger.info(
            "ask answered=%s reason=%s sources=%d cited=%d tokens=%d/%d latency_ms=%.0f",
            result.answerable,
            reason,
            len(sources),
            len(citations),
            usage.input_tokens,
            usage.output_tokens,
            result.latency_ms["total"],
        )
        return result
