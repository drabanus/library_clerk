"""
Abstract LLM backend interface.

All backends (Ollama, Claude, Perplexity) implement this interface so
the classifier can switch between them via config without code changes.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Base class every LLM backend must implement."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._call_count = 0
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Public API (used by PublicationClassifier)
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        format_json: bool = False,
    ) -> str:
        """Generate a text completion with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                text = self._call(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    format_json=format_json,
                )
                self._call_count += 1
                return text.strip()
            except Exception as e:
                last_error = e
                wait = self._backoff(attempt)
                logger.warning(
                    f"{self.name} call failed "
                    f"(attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise ConnectionError(
            f"{self.name} failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Generate a response and parse it as JSON."""
        raw = self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format_json=True,
        )
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for log messages."""
        ...

    @abstractmethod
    def _call(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        format_json: bool,
    ) -> str:
        """Execute a single LLM call (no retry). Return raw text."""
        ...

    def check_models(self, models: list[str]) -> dict[str, bool]:
        """Check model availability. API backends assume all are available."""
        return {m: True for m in models}

    @property
    def stats(self) -> dict:
        return {
            "backend": self.name,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> int:
        """Backoff schedule: 2s, 5s, 10s for API; overridden by Ollama."""
        return [2, 5, 10][min(attempt, 2)]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling common formatting issues."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        code_block = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
        )
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from LLM output: {text[:200]}...")
        return {"raw_response": text, "_parse_error": True}
