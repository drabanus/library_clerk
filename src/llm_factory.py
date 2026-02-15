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


def _resolve_backend_cfg(config: dict[str, Any]) -> tuple[str, dict]:
    """
    Return (backend_name, backend_settings) from the config.

    Supports three layouts:
      1. Nested:   llm.backend + llm.<backend_name>.{...}   (preferred)
      2. Flat:     llm.backend + settings directly in llm    (legacy)
      3. Top-level ollama: section                           (oldest)
    """
    llm_cfg = config.get("llm", {})
    backend_name = llm_cfg.get("backend", "").lower()

    if not backend_name:
        # Oldest layout: top-level "ollama" section, no "llm" at all
        ollama_cfg = config.get("ollama", {})
        return "ollama", ollama_cfg if ollama_cfg else {}

    # Prefer nested sub-section (e.g. llm.claude: {...})
    nested = llm_cfg.get(backend_name)
    if isinstance(nested, dict) and nested:
        return backend_name, nested

    # Fall back to flat layout (keys live directly under llm)
    return backend_name, llm_cfg


def create_llm_backend(config: dict[str, Any]) -> tuple[LLMBackend, str, str]:
    """
    Build an LLM backend from the application config.

    Returns:
        (backend_instance, classifier_model_name, ontology_model_name)
    """
    backend_name, cfg = _resolve_backend_cfg(config)

    defaults = _DEFAULTS.get(backend_name, _DEFAULTS["ollama"])
    classifier_model = cfg.get("classifier_model", defaults["classifier_model"])
    ontology_model = cfg.get("ontology_model", defaults["ontology_model"])

    if backend_name == "ollama":
        from .ollama_client import OllamaClient
        client = OllamaClient(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout=cfg.get("timeout", 120),
            max_retries=cfg.get("max_retries", 3),
            num_ctx=cfg.get("num_ctx", 0),
        )

    elif backend_name == "claude":
        from .claude_backend import ClaudeBackend
        client = ClaudeBackend(
            api_key=cfg.get("api_key") or None,
            max_retries=cfg.get("max_retries", 3),
            max_tokens=cfg.get("max_tokens", 4096),
        )

    elif backend_name == "perplexity":
        from .perplexity_backend import PerplexityBackend
        client = PerplexityBackend(
            api_key=cfg.get("api_key") or None,
            base_url=cfg.get("base_url", "https://api.perplexity.ai"),
            max_retries=cfg.get("max_retries", 3),
            max_tokens=cfg.get("max_tokens", 4096),
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
