from pagewise.retrievers import BM25Retriever


def test_bm25_ranks_matching_text_higher():
    retriever = BM25Retriever()
    retriever.fit([
        "The sky is blue and clear today.",
        "Pricing starts at ten dollars per month.",
        "Cats are small furry animals.",
    ])
    scores = retriever.score("How much does it cost per month?")
    assert scores.argmax() == 1


def test_bm25_state_roundtrip():
    retriever = BM25Retriever()
    retriever.fit(["alpha beta", "gamma delta"])
    state = retriever.state()

    restored = BM25Retriever()
    restored.load_state(state)
    assert (restored.score("alpha") == retriever.score("alpha")).all()


def test_bm25_empty_corpus():
    retriever = BM25Retriever()
    retriever.fit([])
    assert retriever.score("anything").size == 0
