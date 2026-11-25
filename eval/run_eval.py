"""Evaluate Ask Waheed: retrieval quality and answer behaviour.

Usage
  python -m eval.run_eval              # retrieval only (no API key needed)
  python -m eval.run_eval --answers    # also call the LLM and grade answers

Metrics
  retrieval hit@k : the expected section appears in the top-k retrieved chunks
  fact accuracy   : the answer contains every required string
  decline rate    : off-topic / sensitive / injection prompts are declined
  no-hallucination: the answer avoids claims that are not in the knowledge base
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.llm import LLMClient, providers_from_env
from app.prompts import SYSTEM_PROMPT, format_context
from app.rag import Retriever

ROOT = Path(__file__).resolve().parent.parent
DECLINE_MARKERS = re.compile(
    r"(don't have|do not have|can't help|cannot help|can only|only answer|not able|unable|"
    r"questions about waheed|happy to answer|not something i|email waheed)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="call the LLM and grade answers")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cases = [json.loads(l) for l in (ROOT / "eval/cases.jsonl").read_text().splitlines() if l.strip()]
    retriever = Retriever(ROOT / "knowledge", embed_model=None if not args.answers else "BAAI/bge-small-en-v1.5")
    llm = LLMClient(providers_from_env()) if args.answers else None

    hits = graded = passed = 0
    retrieval_cases = [c for c in cases if c["section"]]
    print(f"Retrieval mode: {retriever.mode}\n")
    for c in cases:
        chunks = retriever.search(c["q"], top_k=args.k)
        hit = None
        if c["section"]:
            hit = any(c["section"].lower() in ch.title.lower() for ch in chunks)
            hits += hit
        line = f"[{'HIT ' if hit else 'MISS' if hit is False else ' -- '}] {c['q']}"

        if llm:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(chunks))},
                    {"role": "user", "content": c["q"]}]
            ans = llm.chat(msgs)
            if c["type"] == "fact":
                ok = all(s.lower() in ans.lower() for s in c["must_include"])
            elif c["type"] == "decline":
                ok = bool(DECLINE_MARKERS.search(ans))
            else:
                ok = not any(s.lower() in ans.lower() for s in c.get("must_not_include", []))
            graded += 1
            passed += ok
            line += f"\n       {'PASS' if ok else 'FAIL'} ({c['type']}): {ans[:160]!r}"
        print(line)

    print(f"\nRetrieval hit@{args.k}: {hits}/{len(retrieval_cases)} = {hits / len(retrieval_cases):.0%}")
    if graded:
        print(f"Answer pass rate: {passed}/{graded} = {passed / graded:.0%}")


if __name__ == "__main__":
    main()
