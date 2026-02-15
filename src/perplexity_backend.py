"""
Perplexity API backend.

Perplexity exposes an OpenAI-compatible chat completions endpoint, so
we use the openai Python SDK pointed at their base URL.

Requires PERPLEXITY_API_KEY env var or api_key in the config.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _lazy_import():
    """Import openai at call time so the dep is optional."""
    try:
        import openai
        return openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for the Perplexity backend. "
            "Install it with: pip install openai"
        )


from .llm_backend import LLMBackend

_PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class PerplexityBackend(LLMBackend):
    """Perplexity API backend (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _PERPLEXITY_BASE_URL,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ):
        super().__init__(max_retries=max_retries)
        openai = _lazy_import()

        resolved_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY env var "
                "or provide api_key in the config."
            )
        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )
        self._max_tokens = max_tokens
        logger.info("Perplexity backend initialised")

    @property
    def name(self) -> str:
        return "Perplexity"

    def _call(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        format_json: bool,
    ) -> str:
        messages = []
        sys_text = system or ""
        if format_json:
            sys_text = (sys_text.rstrip() +
                        "\n\nYou MUST respond with valid JSON only.")
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
        }

        response = self._client.chat.completions.create(**kwargs)

        # Accumulate token usage
        if response.usage:
            self._total_tokens += response.usage.completion_tokens or 0

        return response.choices[0].message.content

    @property
    def stats(self) -> dict:
        base = super().stats
        base["compute"] = "Perplexity API"
        return base
