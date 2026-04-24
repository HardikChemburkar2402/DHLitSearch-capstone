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
    """Wrapper for the Google GenAI client (Vertex AI preferred, AI Studio fallback)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        from google import genai

        # If GOOGLE_CLOUD_PROJECT is set, talk to Gemini through Vertex AI
        # (charged to the cloud project + free credits). Otherwise fall back
        # to the API-key path (AI Studio).
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_AI_LOCATION") or "us-central1"
        prefer_vertex = os.getenv("GEMINI_PREFER_VERTEXAI", "1").strip().lower() not in {"0", "false", "no"}

        self._genai = genai
        self._project = project
        self._location = location
        self._provider = "aistudio"

        try:
            if prefer_vertex and project:
                self._client = genai.Client(vertexai=True, project=project, location=location)
                self._provider = "vertexai"
            else:
                self._client = genai.Client()
                self._provider = "aistudio"
        except Exception:
            # Vertex init can fail for auth / quota reasons — take the API-key
            # path instead so the UI still has a working LLM.
            self._client = genai.Client()
            self._provider = "aistudio"

    def provider_label(self) -> str:
        if self._provider == "vertexai":
            return "Vertex AI"
        return "AI Studio (API key)"

    def _switch_to_aistudio(self) -> None:
        self._client = self._genai.Client()
        self._provider = "aistudio"

    def generate(self, prompt: str) -> LLMResult:
        # If Vertex fails at runtime (auth/quota), fall back to AI Studio once.
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt)
                text = getattr(response, "text", None) or ""
                if not text.strip():
                    raise LLMProviderError("Gemini returned empty text.")
                return LLMResult(text=text, provider=f"gemini-{self._provider}", model=self.model)
            except Exception as e:
                if attempt == 0 and self._provider == "vertexai":
                    self._switch_to_aistudio()
                    continue
                raise LLMProviderError(str(e)) from e


class OllamaLLM:
    """Local LLM via Ollama — needs the service running and the model pulled."""

    def __init__(self, model: str = "llama3", host: Optional[str] = None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST")
        # lazy import so the project runs even without ollama installed
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

