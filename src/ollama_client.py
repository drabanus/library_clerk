"""
Ollama LLM backend with GPU detection and timeout auto-scaling.

Provides local inference via Ollama for:
  1. ministral-3:3b  -> fast classification, keyword extraction
  2. qwen2.5:7b      -> structured ontology generation
"""

import logging
from typing import Optional

import ollama

from .gpu_probe import probe_gpu, get_ollama_options
from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)


class OllamaClient(LLMBackend):
    """Ollama local-inference backend with GPU auto-detection."""

    # CPU inference is much slower; scale the read timeout so requests
    # don't time out before the model finishes generating.
    _CPU_TIMEOUT_MULTIPLIER = 5

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        super().__init__(max_retries=max_retries)
        self.base_url = base_url

        # Probe GPU and derive Ollama runtime options
        self.gpu_info = probe_gpu()
        self._hw_options = get_ollama_options(self.gpu_info)

        # Widen the read timeout for CPU-only inference so that large
        # models (e.g. deepseek-coder:6.7b) have time to finish.
        if not self.gpu_info.has_gpu:
            timeout = timeout * self._CPU_TIMEOUT_MULTIPLIER
            logger.info(
                f"CPU-only mode: read timeout raised to {timeout}s"
            )
        self.timeout = timeout

        import httpx
        http_timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout),
            write=10.0,
            pool=10.0,
        )
        self._client = ollama.Client(host=base_url, timeout=http_timeout)

    @property
    def name(self) -> str:
        return "Ollama"

    def _call(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        format_json: bool,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {"temperature": temperature, **self._hw_options}

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "options": options,
        }
        if format_json:
            kwargs["format"] = "json"

        response = self._client.chat(**kwargs)

        if "eval_count" in response:
            self._total_tokens += response.get("eval_count", 0)

        return response["message"]["content"]

    def _backoff(self, attempt: int) -> int:
        """Longer backoff for Ollama: model loading can take a while."""
        return [10, 30, 60][min(attempt, 2)]

    def check_models(self, models: list[str]) -> dict[str, bool]:
        """Check which models are available locally."""
        try:
            response = self._client.list()
            available_names = set()
            for m in response.models:
                name = m.model or ""
                available_names.add(name)
                base_name = name.split(":")[0]
                available_names.add(base_name)

            logger.debug(f"Available Ollama models: {sorted(available_names)}")
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
        base = super().stats
        base["compute"] = self.gpu_info.summary
        return base
