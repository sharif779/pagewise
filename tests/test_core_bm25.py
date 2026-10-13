from pagewise import DocQA


def test_ingest_and_ask_bm25(sample_path):
    qa = DocQA(mode="bm25")
    count = qa.ingest(sample_path)
    assert count > 0

    result = qa.ask("How much does the Pro plan cost?")
    assert result.found
    assert "29" in result.answer
    assert result.source == "document"


def test_ask_returns_not_found_for_unrelated_question(sample_path):
    qa = DocQA(mode="bm25")
    qa.ingest(sample_path)
    result = qa.ask("What is the airspeed velocity of an unladen swallow?", threshold=0.9)
    assert not result.found


def test_save_and_load_roundtrip(tmp_path, sample_path):
    qa = DocQA(mode="bm25")
    qa.ingest(sample_path)
    index_path = tmp_path / "index.pkl"
    qa.save(str(index_path))

    loaded = DocQA.load(str(index_path))
    result = loaded.ask("How much does the Pro plan cost?")
    assert result.found
    assert "29" in result.answer


def test_faq_takes_priority(sample_path):
    qa = DocQA(mode="bm25")
    qa.ingest(sample_path)
    qa.add_faq("What is your refund policy?", "We offer a 30-day money-back guarantee.")

    result = qa.ask("What is your refund policy?")
    assert result.source == "faq"
    assert "money-back" in result.answer


def test_unpunctuated_lines_do_not_merge_into_one_answer(tmp_path):
    doc = tmp_path / "facts.txt"
    doc.write_text(
        "Acme is a widget company.\nContact email is info@acme.com\nFounded 2006",
        encoding="utf-8",
    )
    qa = DocQA(mode="bm25")
    qa.ingest(str(doc))

    result = qa.ask("What is the contact email?")
    assert result.found
    assert result.answer == "Contact email is info@acme.com"
    assert "2006" not in result.answer


def test_size_limit_enforced(tmp_path):
    import pytest

    from pagewise.parsers import FileTooLarge

    big_file = tmp_path / "big.txt"
    big_file.write_text("word " * 2_000_000)  # well over 5 MB

    qa = DocQA(mode="bm25")
    with pytest.raises(FileTooLarge):
        qa.ingest(str(big_file), max_mb=5)
