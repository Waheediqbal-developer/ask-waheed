# Ask Waheed: a RAG chatbot for my portfolio

Recruiters can ask my portfolio questions in plain English ("What did he do at Turing?", "Has he deployed computer vision models?") and get short answers grounded in my actual experience, with the source sections shown.

**Live:** the chat on [my portfolio](https://waheediqbal-developer.github.io) · **API docs:** https://ask-waheed.vercel.app/docs

## How it works

```
question ──► hybrid retrieval ─────────────► prompt with context ──► LLM ──► answer + sources
             ├─ BM25 (keywords)                (strict grounding     (Groq, Gemini
             └─ BGE-small embeddings            rules)                fallback)
                  merged with Reciprocal Rank Fusion
```

- **Knowledge base:** markdown files in `knowledge/`, split at headings so every chunk has a meaningful title (e.g. *Experience > Turing*) that is shown as the citation.
- **Hybrid retrieval:** BM25 catches exact terms like "CGPA" or "YOLO"; dense embeddings (BAAI/bge-small-en-v1.5 via fastembed, ONNX on CPU) catch paraphrases like "where is he employed". Rankings are merged with Reciprocal Rank Fusion. If the embedding model is unavailable, the service falls back to BM25 instead of failing.
- **Follow-up questions:** the previous user question is added to the retrieval query, so "what tools did he use there?" still finds the right section.
- **Grounding and safety:** the system prompt restricts answers to retrieved context, declines off-topic requests and salary questions, and ignores injected instructions.
- **Provider fallback:** any OpenAI-compatible API works. Groq is tried first, then Gemini, both on free tiers.
- **Serverless deployment:** runs on Vercel's Python runtime with zero configuration (`app/main.py` exposes the FastAPI `app`); the same code also runs in Docker.
- **Abuse protection:** per-visitor rate limiting, input length limits and a CORS allow-list.

## Evaluation

`eval/cases.jsonl` holds 17 test cases in three categories: facts, declines (off-topic, salary, prompt injection) and hallucination traps ("Did he work at OpenAI?").

```bash
python -m eval.run_eval             # retrieval hit@4 (no API key needed)
python -m eval.run_eval --answers   # also grades LLM answers
```

| Metric | Result |
|---|---|
| Retrieval hit@4 (BM25 only) | 13/13 (100%) |
| Answer pass rate | *run with `--answers` and fill in* |

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...            # free key from console.groq.com
uvicorn app.main:app --reload --port 7860   # API docs at http://localhost:7860/docs
curl -X POST localhost:7860/api/chat -H "Content-Type: application/json" \
     -d '{"message": "What did Waheed do at Turing?"}'
pytest -q
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` / `GROQ_MODEL` | – / `openai/gpt-oss-120b` | Primary LLM |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | – / `gemini-2.5-flash` | Fallback LLM |
| `ALLOWED_ORIGINS` | GitHub Pages site + localhost | CORS allow-list |
| `RATE_LIMIT_PER_MIN` | `8` | Questions per visitor per minute |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Empty string = BM25 only |

## Stack
Python · FastAPI · rank-bm25 · fastembed (ONNX) · NumPy · Vercel · Docker · Groq / Gemini
