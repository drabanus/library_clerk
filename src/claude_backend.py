"""
Anthropic Claude API backend.

Uses the Anthropic Python SDK. Requires ANTHROPIC_API_KEY env var or
api_key in the config.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _lazy_import():
    """Import anthropic at call time so the dep is optional."""
    try:
        import anthropic
        return anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for the Claude backend. "
            "Install it with: pip install anthropic"
        )


from .llm_backend import LLMBackend


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ):
        super().__init__(max_retries=max_retries)
        anthropic = _lazy_import()

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or provide api_key in the config."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._max_tokens = max_tokens
        logger.info("Claude backend initialised")

    @property
    def name(self) -> str:
        return "Claude"

    def _call(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        format_json: bool,
    ) -> str:
        if format_json and system:
            system = system.rstrip() + "\n\nYou MUST respond with valid JSON only."

        kwargs: dict = {
            "model": model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        # Accumulate token usage
        if hasattr(response, "usage") and response.usage:
            self._total_tokens += response.usage.output_tokens

        return response.content[0].text

    @property
    def stats(self) -> dict:
        base = super().stats
        base["compute"] = "Anthropic API"
        return base
