from pagewise.chunker import make_chunks, split_sentences


def test_split_sentences_basic():
    sentences = split_sentences("Hello there. How are you? I am fine!")
    assert sentences == ["Hello there.", "How are you?", "I am fine!"]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_make_chunks_overlap():
    text = " ".join(f"Sentence {i}." for i in range(10))
    chunks = make_chunks([(1, text)], window=4, overlap=1)
    assert len(chunks) > 1
    assert all(c.page == 1 for c in chunks)
    # consecutive chunks share the overlap sentence
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail in chunks[1].text


def test_make_chunks_rejects_bad_window():
    import pytest

    with pytest.raises(ValueError):
        make_chunks([(1, "a. b. c.")], window=2, overlap=2)


def test_split_sentences_treats_line_breaks_as_boundaries():
    text = "Contact email is info@acme.com\nFounded 2006"
    assert split_sentences(text) == ["Contact email is info@acme.com", "Founded 2006"]


def test_make_chunks_preserves_individual_sentences():
    text = "Contact email is info@acme.com\nFounded 2006"
    chunks = make_chunks([(1, text)], window=4, overlap=1)
    assert len(chunks) == 1
    assert chunks[0].sentences == ("Contact email is info@acme.com", "Founded 2006")
