"""Minimal client for any OpenAI-compatible chat completions API.

Groq and Google Gemini both expose OpenAI-compatible endpoints with free
tiers, so one small client covers both. Providers are tried in order; if
the first fails (rate limit, outage), the next one answers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    extra: dict | None = None  # provider-specific request fields


def _key(name: str) -> str:
    """Read an API key, tolerating quotes/spaces pasted around it (common on Windows)."""
    return os.getenv(name, "").strip().strip('"').strip("'").strip()


def _reasoning_extra(model: str) -> dict:
    # gpt-oss models "think" before answering. Low effort keeps answers fast and
    # stops the thinking from using up the token budget meant for the answer.
    return {"reasoning_effort": "low"} if "gpt-oss" in model else {}


def providers_from_env() -> list[Provider]:
    found = []
    if _key("GROQ_API_KEY"):
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        found.append(Provider("groq", "https://api.groq.com/openai/v1",
                              _key("GROQ_API_KEY"), model, _reasoning_extra(model)))
    if _key("GEMINI_API_KEY"):
        found.append(Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                              _key("GEMINI_API_KEY"),
                              os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()))
    return found


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, providers: list[Provider], timeout: float = 30.0):
        self.providers = providers
        self.http = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1500) -> str:
        if not self.providers:
            raise LLMError("No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY.")
        last_error = None
        for p in self.providers:
            try:
                r = self.http.post(
                    f"{p.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {p.api_key}"},
                    json={"model": p.model, "messages": messages, "temperature": temperature,
                          "max_tokens": max_tokens, **(p.extra or {})},
                )
                if r.status_code == 401:
                    raise LLMError(f"{p.name}: API key rejected (401). Check the key has no quotes or spaces.")
                if r.status_code == 404:
                    raise LLMError(f"{p.name}: model '{p.model}' not found (404). "
                                   f"Pick a current model from the provider's model list.")
                r.raise_for_status()
                content = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    raise LLMError(f"{p.name}: empty answer (token budget used up by reasoning?)")
                return content
            except Exception as e:
                log.warning("Provider %s failed: %s", p.name, e)
                last_error = e
        raise LLMError(f"All providers failed: {last_error}")
