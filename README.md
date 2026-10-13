# pagewise

Answer questions about a PDF, DOCX, or TXT file by finding the passage that already
contains the answer — **no LLM, no API key, no GPU.**

Upload a small product guide (≤5&nbsp;MB) and let visitors ask questions about it.
pagewise finds the most relevant passage with keyword search (BM25) and, optionally,
a small sentence-embedding model (~80&nbsp;MB, CPU-only — not a generative LLM), then
returns the sentence that answers the question, with a page reference.

```python
from pagewise import DocQA

qa = DocQA()
qa.ingest("product_guide.pdf")
result = qa.ask("How much does the Pro plan cost?")
print(result.answer, "-> page", result.page)
```

## Why extractive, not generative

Without a generative model, pagewise can't write a new answer — it can only point to
text that already exists in your document. For product FAQs and overviews that's
usually exactly what you want: the answer is always traceable to a page, and it can
never hallucinate a feature your product doesn't have.

## Install

```bash
pip install pagewise                # BM25 keyword search only, no ML deps
pip install pagewise[semantic]      # + MiniLM sentence embeddings (handles paraphrasing)
pip install pagewise[server]        # + FastAPI server for the /upload and /ask endpoints
```

Model weights are not bundled — the `semantic` extra downloads `all-MiniLM-L6-v2`
from Hugging Face on first use and caches it locally.

## Quick start

```python
from pagewise import DocQA

qa = DocQA()                 # mode="auto": hybrid if sentence-transformers is installed, else BM25
qa.ingest("product_guide.pdf")

result = qa.ask("What integrations do you support?")
print(result.answer)         # the matching sentence
print(result.page)           # where it came from
print(result.found)          # False if nothing matched above the threshold
```

Save and reload an index without re-parsing the document:

```python
qa.save("product_guide.index.pkl")
qa = DocQA.load("product_guide.index.pkl")
```

### Manual FAQ overrides

Pin exact answers to your most common questions — these are checked first, before
document retrieval:

```python
qa.add_faq("What is your refund policy?", "We offer a 30-day money-back guarantee.")
```

## CLI

```bash
pagewise ingest product_guide.pdf --index product_guide.pkl
pagewise ask "How much does the Pro plan cost?" --index product_guide.pkl
pagewise serve --index product_guide.pkl --port 8000
```

## Run the server

```bash
pip install pagewise[server]
pagewise serve --port 8000
```

```bash
curl -F "file=@product_guide.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much does the Pro plan cost?"}'
```

The server is single-tenant by default (one document at a time). For multiple
tenants, run one process per tenant with its own `--index` path.

## Embed the chat widget

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/sharif779/pagewise/widget/chat.css">
<script src="https://cdn.jsdelivr.net/gh/sharif779/pagewise/widget/chat.js" data-api="https://your-api.example.com"></script>
```

This adds a floating chat bubble that calls your server's `/ask` endpoint. No
build step, no framework dependency.

## How it works

1. **Parse** — extract text (and page numbers, for PDFs) with `pypdf` or `python-docx`.
2. **Chunk** — split into overlapping 3–5 sentence windows, tagged with their source page.
3. **Index** — score chunks with BM25 (exact keyword/model-number matches) and,
   optionally, MiniLM sentence embeddings (catches paraphrases like "cost" vs. "price").
   A hybrid retriever combines both.
4. **Answer** — within the best-matching chunks, pick the single sentence that best
   matches the question, and return it with its page number.
5. **Threshold** — if the best match score is too low, respond with
   "I couldn't find that in the document" instead of guessing.

## Limitations

- **It can't combine information.** If the answer spans two sections, you get the
  best single match, not a synthesis. `result.context` returns the full chunk in
  case the surrounding text helps.
- **Scanned PDFs have no text layer.** OCR (e.g. `pytesseract`) isn't included.
- **Tables extract poorly** with `pypdf`. For table-heavy documents, consider
  pre-processing with `pdfplumber`.
- **Answers are extracted, not composed** — expect a sentence lifted from the
  document, not a conversational rewrite.

## License

MIT — see [LICENSE](LICENSE).
