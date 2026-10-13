"""Minimal end-to-end example. Run with: python examples/quickstart.py"""

from pathlib import Path

from pagewise import DocQA

SAMPLE = Path(__file__).parent.parent / "tests" / "sample.txt"


def main() -> None:
    qa = DocQA()  # mode="auto": uses semantic+BM25 if sentence-transformers is installed
    chunks = qa.ingest(str(SAMPLE))
    print(f"Indexed {chunks} chunks (mode={qa.mode})\n")

    for question in [
        "How much does the Pro plan cost?",
        "What integrations are supported?",
        "Is my data encrypted?",
        "What is the capital of France?",
    ]:
        result = qa.ask(question)
        print(f"Q: {question}")
        print(f"A: {result.answer}", f"(page {result.page})" if result.found else "")
        print()


if __name__ == "__main__":
    main()
