"""Offline evaluation: ``python -m docs_assistant.eval``.

Measures, on a labelled question set:
- retrieval hit-rate@k and MRR (is an expected document among the passages given to the model?)
- refusal correctness (answer when the docs cover it, refuse when they don't)
- expected facts present in the answer, and whether the answer cites an expected document
- latency and estimated cost
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .answering import DocsAssistant
from .config import Settings, load_dotenv_if_present

_DASHES = str.maketrans({"–": "-", "—": "-", " ": " ", " ": " "})


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    expected_sources: list[str] = field(default_factory=list)
    expected_facts: list[list[str]] = field(default_factory=list)  # each group: any-of


@dataclass
class QuestionOutcome:
    id: str
    question: str
    answerable: bool
    expected_sources: list[str]
    retrieved_docs: list[str]
    hit: bool | None
    reciprocal_rank: float | None
    refused: bool
    refusal_reason: str | None
    refusal_correct: bool
    facts_found: int
    facts_total: int
    cites_expected: bool | None
    cited_docs: list[str]
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


@dataclass
class EvalReport:
    created_at: str
    mode: str  # "full" or "retrieval-only"
    llm_model: str
    embedding_model: str
    top_k: int
    min_similarity: float
    hybrid: bool
    outcomes: list[QuestionOutcome]

    # ---- aggregate metrics ----

    @property
    def answerable(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.answerable]

    @property
    def unanswerable(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if not o.answerable]

    def summary(self) -> dict[str, Any]:
        answerable, unanswerable = self.answerable, self.unanswerable
        hits = sum(1 for o in answerable if o.hit)
        rr = [o.reciprocal_rank or 0.0 for o in answerable]
        answered_ok = sum(1 for o in answerable if o.refusal_correct)
        refused_ok = sum(1 for o in unanswerable if o.refusal_correct)
        answered = [o for o in answerable if not o.refused]
        latencies = [o.latency_ms for o in self.outcomes]
        costs = [o.cost_usd for o in self.outcomes]
        total_cost = None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)
        return {
            "questions": len(self.outcomes),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "hit_rate": hits / len(answerable) if answerable else None,
            "hits": hits,
            "mrr": statistics.fmean(rr) if rr else None,
            "refusal_correct": answered_ok + refused_ok,
            "refusal_accuracy": (answered_ok + refused_ok) / len(self.outcomes)
            if self.outcomes
            else None,
            "answered_correctly": answered_ok,
            "refused_correctly": refused_ok,
            "facts_found": sum(o.facts_found for o in answered),
            "facts_total": sum(o.facts_total for o in answerable),
            "cites_expected": sum(1 for o in answered if o.cites_expected),
            "answered": len(answered),
            "latency_avg_ms": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "input_tokens": sum(o.input_tokens for o in self.outcomes),
            "output_tokens": sum(o.output_tokens for o in self.outcomes),
            "total_cost_usd": total_cost,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReport:
        fields = {k: v for k, v in data.items() if k not in {"summary", "outcomes"}}
        return cls(**fields, outcomes=[QuestionOutcome(**o) for o in data["outcomes"]])


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    questions = []
    for item in items:
        facts = [[f] if isinstance(f, str) else list(f) for f in item.get("expected_facts", [])]
        questions.append(
            EvalQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                answerable=bool(item.get("answerable", True)),
                expected_sources=list(item.get("expected_sources", [])),
                expected_facts=facts,
            )
        )
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids must be unique")
    return questions


def _normalise(text: str) -> str:
    return " ".join(text.translate(_DASHES).lower().split())


def facts_present(answer: str, groups: list[list[str]]) -> int:
    """Count fact groups with at least one alternative contained in the answer.

    >>> facts_present("Delivery takes 3–5 business days.", [["3-5", "3 to 5"], ["€6.90"]])
    1
    """
    text = _normalise(answer)
    return sum(1 for group in groups if any(_normalise(alt) in text for alt in group))


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def evaluate_question(
    assistant: DocsAssistant, question: EvalQuestion, *, retrieval_only: bool
) -> QuestionOutcome:
    if retrieval_only:
        search = assistant.retriever.search(question.question)
        retrieved = _unique([c.chunk.doc_path for c in search.chunks])
        refused = not search.chunks
        answer, reason, cited_docs = "", ("no_relevant_context" if refused else None), []
        latency, tokens_in, tokens_out = 0.0, 0, 0
        cost: float | None = 0.0
    else:
        result = assistant.ask(question.question)
        retrieved = _unique([s.chunk.doc_path for s in result.sources])
        refused, answer, reason = result.refused, result.answer, result.refusal_reason
        cited_docs = _unique([c.doc_path for c in result.citations])
        latency = result.latency_ms["total"]
        tokens_in, tokens_out = result.usage.input_tokens, result.usage.output_tokens
        cost = result.usage.cost_usd

    expected = set(question.expected_sources)
    hit: bool | None = None
    rr: float | None = None
    if question.answerable and expected:
        ranks = [i for i, doc in enumerate(retrieved, start=1) if doc in expected]
        hit = bool(ranks)
        rr = 1.0 / ranks[0] if ranks else 0.0

    answered = not refused
    return QuestionOutcome(
        id=question.id,
        question=question.question,
        answerable=question.answerable,
        expected_sources=question.expected_sources,
        retrieved_docs=retrieved,
        hit=hit,
        reciprocal_rank=rr,
        refused=refused,
        refusal_reason=reason,
        refusal_correct=answered == question.answerable,
        facts_found=facts_present(answer, question.expected_facts) if answered else 0,
        facts_total=len(question.expected_facts) if question.answerable else 0,
        cites_expected=(bool(expected & set(cited_docs)) if answered and expected else None),
        cited_docs=cited_docs,
        answer=answer,
        latency_ms=latency,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=cost,
    )


def run_eval(
    assistant: DocsAssistant, questions: list[EvalQuestion], *, retrieval_only: bool = False
) -> EvalReport:
    outcomes = [evaluate_question(assistant, q, retrieval_only=retrieval_only) for q in questions]
    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="retrieval-only" if retrieval_only else "full",
        llm_model=assistant.llm.model,
        embedding_model=assistant.retriever.embedder.model,
        top_k=assistant.retriever.top_k,
        min_similarity=assistant.retriever.min_similarity,
        hybrid=assistant.retriever.hybrid,
        outcomes=outcomes,
    )


# ---- rendering --------------------------------------------------------------------------


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "n/a"


def _summary_rows(report: EvalReport) -> list[tuple[str, str]]:
    s = report.summary()
    mrr = f"{s['mrr']:.2f}" if s["mrr"] is not None else "n/a"
    rows = [
        (
            f"Retrieval hit-rate@{report.top_k} (answerable questions)",
            _pct(s["hits"], s["answerable"]),
        ),
        ("Mean reciprocal rank of the expected document", mrr),
        (
            "Refusal correctness (answer vs. “I don't know”)",
            _pct(s["refusal_correct"], s["questions"]),
        ),
        ("· answerable questions answered", _pct(s["answered_correctly"], s["answerable"])),
        ("· unanswerable questions refused", _pct(s["refused_correctly"], s["unanswerable"])),
    ]
    if report.mode == "full":
        cost = s["total_cost_usd"]
        per_question = cost / s["questions"] if cost is not None and s["questions"] else None
        latency = " / ".join(
            f"{s[key] / 1000:.2f} s"
            for key in ("latency_avg_ms", "latency_p50_ms", "latency_max_ms")
        )
        cost_text = (
            f"${cost:.4f} / ${per_question:.5f}"
            if cost is not None and per_question is not None
            else "n/a"
        )
        rows += [
            ("Expected facts present in answers", _pct(s["facts_found"], s["facts_total"])),
            ("Answers citing an expected document", _pct(s["cites_expected"], s["answered"])),
            ("Latency per question (avg / p50 / max)", latency),
            ("Tokens (input / output)", f"{s['input_tokens']:,} / {s['output_tokens']:,}"),
            ("Estimated cost (total / per question)", cost_text),
        ]
    return rows


def _status(o: QuestionOutcome) -> str:
    if o.answerable:
        return "answered" if not o.refused else f"refused ({o.refusal_reason})"
    return f"refused ({o.refusal_reason})" if o.refused else "answered"


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Run: {report.created_at} · mode: **{report.mode}**",
        f"- Models: `{report.llm_model}` (answers) · `{report.embedding_model}` (embeddings)",
        f"- Retrieval: top-k {report.top_k}, min similarity {report.min_similarity:.2f}, "
        f"{'hybrid (vector + BM25, RRF)' if report.hybrid else 'vector only'}",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in _summary_rows(report)]
    lines += [
        "",
        "## Per question",
        "",
        "| # | Question | Expected | Top retrieved | Hit | Outcome | Correct | Facts | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in report.outcomes:
        expected = ", ".join(f"`{d}`" for d in o.expected_sources) or "— (not in docs)"
        top = f"`{o.retrieved_docs[0]}`" if o.retrieved_docs else "—"
        hit = "—" if o.hit is None else ("yes" if o.hit else "no")
        facts = f"{o.facts_found}/{o.facts_total}" if o.facts_total else "—"
        question = o.question.replace("|", "\\|")
        lines.append(
            f"| {o.id} | {question} | {expected} | {top} | {hit} | {_status(o)} | "
            f"{'✅' if o.refusal_correct else '❌'} | {facts} | {o.latency_ms / 1000:.2f} s |"
        )
    if report.mode == "full":
        lines += ["", "## Answers", ""]
        for o in report.outcomes:
            answer = o.answer.replace("\n", " ")
            lines += [f"**{o.id}. {o.question}**", "", f"> {answer}", ""]
    return "\n".join(lines).rstrip() + "\n"


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Docs Assistant — evaluation report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{font_face}
{css}</style></head><body><div class="wrap">
<h1>Evaluation report</h1>
<p class="sub">{subtitle}</p>
<div class="tiles">{tiles}</div>
<h2>Per question</h2><table class="questions"><thead><tr><th>#</th><th>Question</th>
<th>Expected source</th><th>Top retrieved</th><th>Hit</th><th>Outcome</th><th>Facts</th>
<th class="num">Latency</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>All metrics</h2><table class="summary"><tbody>{summary}</tbody></table>
</div></body></html>
"""


def _tiles(report: EvalReport) -> list[tuple[str, str, str]]:
    s = report.summary()
    tiles = [
        (f"Hit-rate@{report.top_k}", f"{s['hits']}/{s['answerable']}", "expected doc retrieved"),
        (
            "Refusal correctness",
            f"{s['refusal_correct']}/{s['questions']}",
            "answered vs. “I don't know”",
        ),
    ]
    if report.mode == "full":
        cost = s["total_cost_usd"]
        tiles += [
            ("Facts present", f"{s['facts_found']}/{s['facts_total']}", "numbers & deadlines"),
            (
                "Avg latency",
                f"{s['latency_avg_ms'] / 1000:.1f} s",
                f"p50 {s['latency_p50_ms'] / 1000:.1f} s · max {s['latency_max_ms'] / 1000:.1f} s",
            ),
            (
                "Total cost",
                f"${cost:.4f}" if cost is not None else "n/a",
                f"{s['questions']} questions",
            ),
        ]
    return tiles


def render_html(report: EvalReport, font_path: Path | None = None) -> str:
    """A self-contained, styled HTML version of the report (font embedded if available)."""
    esc = html.escape
    font_face = ""
    if font_path is not None and font_path.is_file():
        encoded = base64.b64encode(font_path.read_bytes()).decode()
        font_face = (
            "@font-face{font-family:Inter;font-weight:100 900;font-display:swap;"
            f"src:url(data:font/woff2;base64,{encoded}) format('woff2');}}"
        )
    tiles = "".join(
        f'<div class="tile"><div class="t-label">{esc(label)}</div>'
        f'<div class="t-value">{esc(value)}</div><div class="t-sub">{esc(sub)}</div></div>'
        for label, value, sub in _tiles(report)
    )
    summary = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(value)}</td></tr>"
        for name, value in _summary_rows(report)
    )
    rows = []
    for o in report.outcomes:
        expected = ", ".join(o.expected_sources) or "not in docs"
        top = o.retrieved_docs[0] if o.retrieved_docs else "—"
        hit = "—" if o.hit is None else ("yes" if o.hit else "no")
        badge = "ok" if o.refusal_correct else "bad"
        facts = f"{o.facts_found}/{o.facts_total}" if o.facts_total else "—"
        facts_class = "miss" if o.facts_found < o.facts_total and not o.refused else ""
        kind = "" if o.answerable else ' <span class="pill">unanswerable</span>'
        rows.append(
            f"<tr><td class='mono'>{esc(o.id)}</td><td>{esc(o.question)}{kind}</td>"
            f"<td class='mono small'>{esc(expected)}</td><td class='mono small'>{esc(top)}</td>"
            f"<td>{hit}</td><td><span class='badge {badge}'>{esc(_status(o))}</span></td>"
            f"<td class='{facts_class}'>{facts}</td>"
            f"<td class='num'>{o.latency_ms / 1000:.2f} s</td></tr>"
        )
    retrieval = "hybrid: vector + BM25, RRF" if report.hybrid else "vector only"
    subtitle = (
        f"{esc(report.created_at)} · mode <b>{esc(report.mode)}</b> · "
        f"<code>{esc(report.llm_model)}</code> + <code>{esc(report.embedding_model)}</code> · "
        f"top-k {report.top_k}, min similarity {report.min_similarity:.2f}, {esc(retrieval)}"
    )
    css = (Path(__file__).parent / "static" / "report.css").read_text(encoding="utf-8")
    return _REPORT_TEMPLATE.format(
        font_face=font_face,
        css=css,
        subtitle=subtitle,
        tiles=tiles,
        summary=summary,
        rows="".join(rows),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docs_assistant.eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    parser.add_argument("--out", type=Path, default=Path("eval/report.md"))
    parser.add_argument("--json", type=Path, default=Path("eval/report.json"))
    parser.add_argument("--html", type=Path, default=Path("eval/report.html"))
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip answer generation (no chat-completion calls); measures retrieval only",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        metavar="REPORT_JSON",
        help="re-render the Markdown/HTML from a saved JSON report (no API calls)",
    )
    args = parser.parse_args(argv)

    if args.from_json:
        report = EvalReport.from_dict(json.loads(args.from_json.read_text(encoding="utf-8")))
    else:
        load_dotenv_if_present()
        from .services import build_services

        services = build_services(Settings.from_env())
        if services.store.stats()["chunks"] == 0:
            print("The index is empty: run `docs-assistant ingest` first.", file=sys.stderr)
            return 2
        questions = load_questions(args.questions)
        report = run_eval(services.assistant, questions, retrieval_only=args.retrieval_only)

    font = Path(__file__).parent / "static" / "fonts" / "inter-latin-wght-normal.woff2"
    outputs = [(args.out, render_markdown(report)), (args.html, render_html(report, font))]
    if not args.from_json:
        outputs.append(
            (args.json, json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n")
        )
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    for name, value in _summary_rows(report):
        print(f"{name:<55} {value}")
    print("\nReport written to: " + ", ".join(str(path) for path, _ in outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
