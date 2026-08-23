# Project assistant

Grounded question-answering over DeepSentinel's own documentation. Ask it about
the architecture, the GraphSAGE results and how they were measured, the API
contract, or the dataset, and it answers from the repository's markdown — with
citations.

Built for the research context: an ungrounded claim about your own results is
worse than no answer, so the assistant never speaks without sources and says
"not in the documentation" rather than guessing.

## How it works

```
question ─▶ BM25 over project docs ─▶ top passages ─▶ LLM (grounded) ─▶ answer + citations
                                                   └▶ no LLM? passages verbatim
```

| File | Role |
|---|---|
| `knowledge.py` | Discovers markdown across all four components, splits on headings, keeps the heading path for citations |
| `retriever.py` | BM25 with a heading boost and an informative-term gate |
| `service.py` | Answer synthesis; reuses the fusion engine's LLM backend |
| `router.py` | `POST /api/chat`, `GET /api/chat/health`, `GET /api/chat/suggestions` |

**Why BM25 rather than embeddings.** The corpus is ~20 documents of
domain-specific prose whose discriminating terms are exact — `PR-AUC`,
`isotonic`, `stage 3c`, `NOT_APPLICABLE`, `0.4056`. Lexical matching handles
those directly with no model download, no vector store and no warm-up, and
answers in milliseconds. `sentence-transformers` and Chroma are already project
dependencies if semantic recall is ever wanted; `Retriever` is the seam.

**Two guards worth knowing.** Superseded documents (the pre-leakage-fix reports)
stay indexed but are down-weighted, so questions about results return the
current numbers rather than the retracted ones. And a question whose
distinctive words are absent from the corpus — "what is the capital of France" —
is rejected rather than answered from stopword noise.

## Mounting

Already wired into `backend/main.py`:

```python
from chatbot import router as chatbot_router
app.include_router(chatbot_router)
```

The import is guarded, so a problem in the assistant degrades to a warning
rather than taking the API down.

## Running

Mounted in the main backend, it is available wherever that runs. To run it
alone (no other backend dependencies needed):

```bash
pip install fastapi uvicorn pydantic
python serve_chatbot.py          # http://localhost:8100
```

Point the frontend at it with `VITE_API_URL=http://localhost:8100` if you run it
standalone; no change is needed when it is mounted in the main backend.

## LLM configuration

The assistant reuses `backend.llm.forensic_reporter.create_llm_backend()`, so it
inherits the project's existing configuration and both providers:

- **Gemini** — set `GEMINI_API_KEY`, or `[secrets] gemini_api_key` in `config.ini`
- **Ollama** — set `[llm] provider = ollama` for free local inference

With neither configured the assistant answers **extractively**: it returns the
retrieved passages verbatim with citations. Less fluent, never wrong, and it
still demonstrates with no internet and no API key — which matters in a
presentation room.

Check which mode is active:

```bash
curl localhost:8000/api/chat/health
```

## API

```bash
curl -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which novelty actually improves accuracy?"}'
```

```json
{
  "answer": "...",
  "sources": ["GraphSage/docs/system_walkthrough.md › 13. Results › 13.4 Finding 2 ..."],
  "grounded": true,
  "confident": true
}
```

`grounded` is false when the answer is extractive rather than LLM-written.
`confident` is false when retrieval found nothing relevant — the UI can then
avoid presenting the reply as an answer.

## Extending the knowledge base

Add a glob to `INCLUDE_GLOBS` in `knowledge.py`. The corpus is rebuilt at
startup, so new documentation is picked up on the next restart with no
re-indexing step.
