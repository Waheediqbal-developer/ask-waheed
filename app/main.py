"""Ask Waheed: FastAPI service answering questions about Waheed Iqbal's work.

Endpoints
  GET  /health    -> status and retrieval mode
  POST /api/chat  -> {"message": str, "history": [{"role","content"}]}
                     returns {"answer": str, "sources": [str]}
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm import LLMClient, LLMError, providers_from_env
from .prompts import SYSTEM_PROMPT, format_context
from .rag import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ask-waheed")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", Path(__file__).resolve().parent.parent / "knowledge"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5") or None
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "https://waheediqbal-developer.github.io,http://localhost:8000,http://127.0.0.1:8000"
).split(",") if o.strip()]
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "8"))
MAX_MESSAGE_CHARS = 500
MAX_HISTORY_TURNS = 6

app = FastAPI(title="Ask Waheed", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)

retriever = Retriever(KNOWLEDGE_DIR, EMBED_MODEL)
llm = LLMClient(providers_from_env())
_hits: dict[str, deque] = defaultdict(deque)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def check_rate_limit(client_id: str) -> None:
    now, window = time.time(), _hits[client_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions in a minute. Wait a moment and ask again.")
    window.append(now)


def client_id(request: Request) -> str:
    # Hosting platforms sit behind a proxy; the first X-Forwarded-For entry is the visitor.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


@app.get("/health")
def health():
    return {"status": "ok", "retrieval": retriever.mode, "chunks": len(retriever.chunks),
            "providers": [p.name for p in llm.providers]}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    check_rate_limit(client_id(request))
    try:
        answer, sources = answer_question(req.message, [t.model_dump() for t in req.history])
    except LLMError as e:
        log.error("LLM failure: %s", e)
        raise HTTPException(503, "The assistant can't reach its language model right now. Try again in a minute.")
    return ChatResponse(answer=answer, sources=sources)


def answer_question(message: str, history: list[dict]) -> tuple[str, list[str]]:
    """The RAG pipeline, used by the JSON API (and any other front end).

    history: [{"role": "user"|"assistant", "content": str}, ...]
    """
    history = [{"role": t["role"], "content": t["content"]} for t in history
               if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)]
    history = history[-MAX_HISTORY_TURNS:]

    # Follow-ups like "what tools did he use there?" need the previous question for retrieval.
    prev_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
    chunks = retriever.search(f"{prev_user} {message}".strip(), top_k=4)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(chunks))}]
    messages += history
    messages.append({"role": "user", "content": message})

    answer = llm.chat(messages)
    return answer, list(dict.fromkeys(c.title for c in chunks))
