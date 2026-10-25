"""Command line: ``docs-assistant ingest | ask | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from .config import Settings, load_dotenv_if_present


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP client debug logs can include request metadata; keep them quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    from .services import build_ingestor

    report = build_ingestor(settings).run(rebuild=args.rebuild)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    from .services import build_services

    result = build_services(settings).assistant.ask(args.question, top_k=args.top_k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(textwrap.fill(result.answer, width=100))
    if result.refused:
        print(f"\n(refused: {result.refusal_reason})")
    for citation in result.citations:
        print(f"  [{citation.number}] {citation.doc_title} › {citation.heading}")
    usage = result.usage
    cost = f"${usage.cost_usd:.5f}" if usage.cost_usd is not None else "n/a"
    print(
        f"\n{result.latency_ms['total']:.0f} ms · {usage.input_tokens}+{usage.output_tokens} "
        f"tokens · {cost}"
    )
    return 0


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-assistant", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index new or changed documents")
    p_ingest.add_argument("--rebuild", action="store_true", help="re-process every document")

    p_ask = sub.add_parser("ask", help="ask one question from the terminal")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true", help="print the full structured result")

    p_serve = sub.add_parser("serve", help="run the API and chat UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    load_dotenv_if_present()
    _configure_logging(args.verbose)
    settings = Settings.from_env()
    handlers = {"ingest": _cmd_ingest, "ask": _cmd_ask, "serve": _cmd_serve}
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
