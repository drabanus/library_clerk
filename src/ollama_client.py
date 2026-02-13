"""
Ollama LLM client with retry logic and structured output parsing.

Provides a clean interface for sequential model invocation:
  1. ministral-3:3b  -> fast classification, keyword extraction
  2. deepseek-coder:6.7b -> structured ontology generation
"""

import json
import time
import logging
import re
from typing import Any, Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper around the Ollama Python client with retry and JSON parsing."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = ollama.Client(host=base_url, timeout=timeout)
        self._call_count = 0
        self._total_tokens = 0

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        format_json: bool = False,
    ) -> str:
        """
        Generate a response from a model.

        Args:
            model: Ollama model name (e.g. "ministral-3:3b")
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature (lower = more deterministic)
            format_json: If True, request JSON output format

        Returns:
            The model's response text.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {"temperature": temperature}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "options": options,
                }
                if format_json:
                    kwargs["format"] = "json"

                response = self._client.chat(**kwargs)
                self._call_count += 1

                content = response["message"]["content"]

                if "eval_count" in response:
                    self._total_tokens += response.get("eval_count", 0)

                return content.strip()

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"Ollama call failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Ollama failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Generate a response and parse it as JSON.
        Falls back to extracting JSON from markdown code blocks.
        """
        raw = self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format_json=True,
        )
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling common formatting issues."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
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

    def check_models(self, models: list[str]) -> dict[str, bool]:
        """Check which models are available locally."""
        try:
            available = self._client.list()
            available_names = set()
            for m in available.get("models", []):
                available_names.add(m["name"])
                # Also add without tag for partial matching
                base_name = m["name"].split(":")[0]
                available_names.add(base_name)
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return {m: False for m in models}

        result = {}
        for model in models:
            result[model] = (
                model in available_names
                or model.split(":")[0] in available_names
            )
        return result

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }
