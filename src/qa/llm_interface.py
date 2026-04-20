from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str


class LLMProviderError(RuntimeError):
    pass


class GeminiLLM:
    """
    Thin wrapper around the Google Generative AI client.
    We keep this isolated so the rest of the app can gracefully fall back when
    Gemini is rate-limited or unavailable.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        # Import lazily so the project can run without Gemini installed/configured.
        from google import genai

        # GEMINI_API_KEY is expected to be present in environment; genai.Client()
        # will pick it up.
        self._client = genai.Client()

    def generate(self, prompt: str) -> LLMResult:
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
            text = getattr(response, "text", None) or ""
            if not text.strip():
                raise LLMProviderError("Gemini returned empty text.")
            return LLMResult(text=text, provider="gemini", model=self.model)
        except Exception as e:
            raise LLMProviderError(str(e)) from e


class OllamaLLM:
    """
    Local LLM via Ollama. Requires the Ollama app/service to be running and the
    model to be pulled (e.g., `ollama pull llama3`).
    """

    def __init__(self, model: str = "llama3", host: Optional[str] = None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST")
        # Import lazily so installs without Ollama still work.
        import ollama

        self._ollama = ollama

    def generate(self, prompt: str) -> LLMResult:
        try:
            kwargs = {}
            if self.host:
                kwargs["host"] = self.host
            resp = self._ollama.generate(model=self.model, prompt=prompt, **kwargs)
            text = (resp or {}).get("response", "") or ""
            if not text.strip():
                raise LLMProviderError("Ollama returned empty text.")
            return LLMResult(text=text, provider="ollama", model=self.model)
        except Exception as e:
            raise LLMProviderError(str(e)) from e

