import pytest

pytest.importorskip("sentence_transformers")

from pagewise import DocQA


def test_hybrid_mode_finds_paraphrased_question(sample_path):
    qa = DocQA(mode="hybrid")
    qa.ingest(sample_path)

    result = qa.ask("What does the Pro subscription cost each month?")
    assert result.found
    assert "29" in result.answer
