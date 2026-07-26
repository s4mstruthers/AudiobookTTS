"""Optional LLM pass that rewrites chapter text for narration.

Fails soft by design: any provider error returns the input text unchanged,
so a dead Ollama server or missing API key never kills a render.
"""

from __future__ import annotations

import os

import httpx

from audiobooktts.config import LLMConfig

SYSTEM_PROMPT = (
    "You prepare book text for text-to-speech narration. Rewrite the user's text "
    "so it reads aloud naturally: expand abbreviations and numerals into words where "
    "a narrator would, remove footnote artifacts, reference markers, and page numbers, "
    "and spell out symbols. Keep the wording otherwise IDENTICAL. Never summarize, "
    "never skip content, never add commentary. Output only the rewritten text."
)

_BLOCK_CHARS = 4000


def _blocks(text: str) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > _BLOCK_CHARS:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}".strip()
    if cur:
        out.append(cur)
    return out


class BaseCleaner:
    def _rewrite(self, block: str) -> str:
        raise NotImplementedError

    def clean(self, text: str) -> str:
        try:
            return "\n\n".join(self._rewrite(b) for b in _blocks(text))
        except Exception:
            return text


class OllamaCleaner(BaseCleaner):
    def __init__(self, cfg: LLMConfig):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.model

    def _rewrite(self, block: str) -> str:
        r = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": block},
                ],
            },
            timeout=300,
        )
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()
        return out if out else block


class AnthropicCleaner(BaseCleaner):
    def __init__(self, cfg: LLMConfig):
        self.model = cfg.model if cfg.model.startswith("claude") else "claude-haiku-4-5-20251001"
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _rewrite(self, block: str) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 8192,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": block}],
            },
            timeout=300,
        )
        r.raise_for_status()
        out = "".join(p["text"] for p in r.json()["content"] if p["type"] == "text").strip()
        return out if out else block


class NoopCleaner(BaseCleaner):
    def clean(self, text: str) -> str:
        return text


def get_cleaner(cfg: LLMConfig) -> BaseCleaner:
    if cfg.provider == "ollama":
        return OllamaCleaner(cfg)
    if cfg.provider == "anthropic":
        return AnthropicCleaner(cfg)
    return NoopCleaner()
