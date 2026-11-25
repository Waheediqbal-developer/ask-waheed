"""Hybrid retrieval over a small markdown knowledge base.

Pipeline:
  1. Split each markdown file into chunks at "## " headings, keeping the
     parent "# " heading as context (e.g. "Experience > Turing ...").
  2. Rank chunks with BM25 (keyword match) and, if available, dense
     embeddings (semantic match).
  3. Merge the two rankings with Reciprocal Rank Fusion (RRF).

Dense embeddings are optional: if fastembed or the model download is not
available, retrieval falls back to BM25 only, so the service never fails
to start because of the embedding model.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be by did do does for from has have he his how i in is it "
    "its me of on or that the their this to was what when where which who why with "
    "you your about tell can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    title: str  # e.g. "Experience > LLM Data and Evaluation Specialist at Turing"
    text: str

    @property
    def indexed_text(self) -> str:
        # Title is included so that queries like "Turing" hit the right chunk.
        return f"{self.title}\n{self.text}"


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        section, sub, buf = "", "", []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                title = f"{section} > {sub}" if sub else section
                chunks.append(Chunk(id=f"{path.stem}#{len(chunks)}", title=title, text=body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                flush(); buf = []
                section, sub = line[2:].strip(), ""
            elif line.startswith("## "):
                flush(); buf = []
                sub = line[3:].strip()
            else:
                buf.append(line)
        flush()
    return chunks


class DenseIndex:
    """Thin wrapper around fastembed (ONNX, CPU-friendly, no torch)."""

    def __init__(self, texts: list[str], model_name: str):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.model = TextEmbedding(model_name=model_name)
        self.matrix = self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def rank(self, query: str) -> list[int]:
        q = self._embed([query])[0]
        return list(np.argsort(-(self.matrix @ q)))


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, knowledge_dir: Path, embed_model: str | None = "BAAI/bge-small-en-v1.5"):
        self.chunks = load_chunks(knowledge_dir)
        if not self.chunks:
            raise ValueError(f"No knowledge chunks found in {knowledge_dir}")
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in self.chunks])
        self.dense: DenseIndex | None = None
        if embed_model:
            try:
                self.dense = DenseIndex([c.indexed_text for c in self.chunks], embed_model)
                log.info("Dense index ready (%s)", embed_model)
            except Exception as e:  # network or model problems: degrade, don't crash
                log.warning("Dense retrieval disabled, using BM25 only: %s", e)

    @property
    def mode(self) -> str:
        return "hybrid" if self.dense else "bm25"

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        tokens = tokenize(query)
        rankings: list[list[int]] = []
        if tokens:
            scores = self.bm25.get_scores(tokens)
            rankings.append([i for i in np.argsort(-scores) if scores[i] > 0])
        if self.dense:
            rankings.append(self.dense.rank(query))
        if not rankings or not any(rankings):
            # Nothing matched: return the profile summary so the model can
            # still give a general answer or say it does not know.
            return self.chunks[:1]
        return [self.chunks[i] for i, _ in rrf(rankings)[:top_k]]
