"""
Factory that instantiates the configured LLM backend.

Reads the top-level ``llm`` key in the config dict.  Falls back to
``ollama`` key for backward compatibility when ``llm`` is absent.

Config examples
---------------

Ollama (default / legacy):
    llm:
      backend: ollama
      base_url: http://localhost:11434
      timeout: 300

Claude:
    llm:
      backend: claude
      api_key: sk-ant-...          # or set ANTHROPIC_API_KEY env var
      classifier_model: claude-sonnet-4-20250514
      ontology_model: claude-sonnet-4-20250514

Perplexity:
    llm:
      backend: perplexity
      api_key: pplx-...            # or set PERPLEXITY_API_KEY env var
      classifier_model: sonar-pro
      ontology_model: sonar-pro
"""

import logging
from typing import Any

from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)

# Default models per backend
_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {
        "classifier_model": "ministral-3:3b",
        "ontology_model": "qwen2.5:7b",
    },
    "claude": {
        "classifier_model": "claude-sonnet-4-20250514",
        "ontology_model": "claude-sonnet-4-20250514",
    },
    "perplexity": {
        "classifier_model": "sonar-pro",
        "ontology_model": "sonar-pro",
    },
}


def create_llm_backend(config: dict[str, Any]) -> tuple[LLMBackend, str, str]:
    """
    Build an LLM backend from the application config.

    Returns:
        (backend_instance, classifier_model_name, ontology_model_name)
    """
    # New-style config: top-level "llm" section
    llm_cfg = config.get("llm", {})
    backend_name = llm_cfg.get("backend", "").lower()

    # Backward-compat: if no "llm" section, fall back to "ollama" section
    if not backend_name:
        ollama_cfg = config.get("ollama", {})
        if ollama_cfg:
            backend_name = "ollama"
            llm_cfg = ollama_cfg
        else:
            backend_name = "ollama"

    defaults = _DEFAULTS.get(backend_name, _DEFAULTS["ollama"])
    classifier_model = llm_cfg.get("classifier_model", defaults["classifier_model"])
    ontology_model = llm_cfg.get("ontology_model", defaults["ontology_model"])

    if backend_name == "ollama":
        from .ollama_client import OllamaClient
        client = OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            timeout=llm_cfg.get("timeout", 120),
            max_retries=llm_cfg.get("max_retries", 3),
        )

    elif backend_name == "claude":
        from .claude_backend import ClaudeBackend
        client = ClaudeBackend(
            api_key=llm_cfg.get("api_key"),
            max_retries=llm_cfg.get("max_retries", 3),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )

    elif backend_name == "perplexity":
        from .perplexity_backend import PerplexityBackend
        client = PerplexityBackend(
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url", "https://api.perplexity.ai"),
            max_retries=llm_cfg.get("max_retries", 3),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )

    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend_name}'. "
            f"Supported: ollama, claude, perplexity"
        )

    logger.info(
        f"LLM backend: {client.name} "
        f"(classifier={classifier_model}, ontology={ontology_model})"
    )
    return client, classifier_model, ontology_model
