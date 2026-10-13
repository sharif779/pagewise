"""Command-line interface: `pagewise ingest|ask|serve`."""

from __future__ import annotations

import argparse
import json
import sys

from .core import DocQA

DEFAULT_INDEX = "pagewise_index.pkl"


def _cmd_ingest(args: argparse.Namespace) -> int:
    qa = DocQA(mode=args.mode)
    count = qa.ingest(args.file, window=args.window, overlap=args.overlap, max_mb=args.max_mb)
    qa.save(args.index)
    print(f"Indexed {count} chunks from {args.file} -> {args.index} (mode={qa.mode})")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    qa = DocQA.load(args.index)
    result = qa.ask(args.question, k=args.k, threshold=args.threshold)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The server requires extra dependencies: pip install pagewise[server]", file=sys.stderr)
        return 1

    from .server import create_app

    app = create_app(index_path=args.index, max_mb=args.max_mb)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagewise", description="Extractive Q&A over a PDF/DOCX/TXT file. No LLM.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Parse and index a document.")
    ingest_p.add_argument("file")
    ingest_p.add_argument("--index", default=DEFAULT_INDEX, help="Where to save the index.")
    ingest_p.add_argument("--mode", default="auto", choices=["auto", "bm25", "semantic", "hybrid"])
    ingest_p.add_argument("--window", type=int, default=4, help="Sentences per chunk.")
    ingest_p.add_argument("--overlap", type=int, default=1, help="Overlapping sentences between chunks.")
    ingest_p.add_argument("--max-mb", type=float, default=5.0, dest="max_mb")
    ingest_p.set_defaults(func=_cmd_ingest)

    ask_p = sub.add_parser("ask", help="Ask a question against a saved index.")
    ask_p.add_argument("question")
    ask_p.add_argument("--index", default=DEFAULT_INDEX)
    ask_p.add_argument("--k", type=int, default=3)
    ask_p.add_argument("--threshold", type=float, default=0.2)
    ask_p.set_defaults(func=_cmd_ask)

    serve_p = sub.add_parser("serve", help="Run the FastAPI server (pip install pagewise[server]).")
    serve_p.add_argument("--index", default=DEFAULT_INDEX)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--max-mb", type=float, default=5.0, dest="max_mb")
    serve_p.set_defaults(func=_cmd_serve)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
