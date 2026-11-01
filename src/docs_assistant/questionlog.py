"""Optional JSONL log of the questions asked through ``POST /ask`` (opt-in, off by default).

One line per question: time, the question, whether it was refused and why, the documents it
cited and how long it took. No IP address, no user identity, no API keys. Questions can contain
personal data, so the file stays wherever ``QUESTION_LOG_PATH`` points (the client's server).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import AskResult

logger = logging.getLogger(__name__)


def build_entry(result: AskResult, *, now: datetime | None = None) -> dict[str, Any]:
    """The log line for one answer.

    >>> from docs_assistant.models import AskResult
    >>> refused = AskResult("Do you ship to Mars?", "I don't know.", False,
    ...                     "no_relevant_context", [], [], None, {"total": 12.34})
    >>> build_entry(refused, now=datetime(2026, 11, 1, 14, 0, tzinfo=UTC))
    {'ts': '2026-11-01T14:00:00+00:00', 'question': 'Do you ship to Mars?', 'refused': True, \
'refusal_reason': 'no_relevant_context', 'cited_docs': [], 'latency_ms': 12.3}
    """
    cited = list(dict.fromkeys(c.doc_path for c in result.citations))
    return {
        "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "question": result.question,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "cited_docs": cited,
        "latency_ms": round(result.latency_ms.get("total", 0.0), 1),
    }


class QuestionLog:
    """Appends one JSON line per question. Disabled when ``path`` is None."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, result: AskResult) -> None:
        """Write the entry; a full disk or a bad path must not break the answer."""
        if self.path is None:
            return
        line = json.dumps(build_entry(result), ensure_ascii=False)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as exc:
            logger.error("Could not write the question log %s: %s", self.path, exc)


def read_entries(path: Path, *, refused_only: bool = False) -> Iterator[dict[str, Any]]:
    """Read logged questions, oldest first. Damaged lines are skipped with a warning.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     log = Path(tmp) / "q.jsonl"
    ...     _ = log.write_text('{"question": "a", "refused": false}\\nnot json\\n'
    ...                        '{"question": "b", "refused": true}\\n', encoding="utf-8")
    ...     [e["question"] for e in read_entries(log)], [e["question"] for e in
    ...      read_entries(log, refused_only=True)]
    (['a', 'b'], ['b'])
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s line %d is not valid JSON; skipped", path, number)
                continue
            if not isinstance(entry, dict):
                continue
            if refused_only and not entry.get("refused"):
                continue
            yield entry
