"""Model presets, decoding defaults, and LM-builder shared across scripts."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


try:
    import dspy
except ModuleNotFoundError as exc:  # pragma: no cover - validated at runtime
    dspy = None  # type: ignore
    DSPY_IMPORT_ERROR: Optional[BaseException] = exc
else:
    DSPY_IMPORT_ERROR = None

# Suppress LiteLLM's verbose error blobs (which can echo request URLs/keys).
try:
    import litellm

    litellm.suppress_debug_info = True
except ModuleNotFoundError:
    pass


def _get_api_key(var_name: str) -> Optional[str]:
    """Return env var, falling back to loading a .env file if available."""

    api_key = os.environ.get(var_name)
    if api_key:
        return api_key
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        return None
    return os.environ.get(var_name)


# ---------------------------------------------------------------------------
# Model presets

PRESET_MAP: Dict[str, Tuple[str, str]] = {
    "qwen3-32b": ("local", "qwen3-32b"),
    "qwen3-14b": ("local", "qwen3-14b"),
    "qwen3-8b": ("local", "qwen3-8b"),
    "qwen3-4b": ("local", "qwen3-4b"),
    "qwen3-1.7b": ("local", "qwen3-1.7b"),
    "qwen3-0.6b": ("local", "qwen3-0.6b"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt5": ("openai", "gpt-5"),
    "claude-sonnet-4.5": ("openrouter", "anthropic/claude-sonnet-4.5"),
    "gemini-2.5-pro": ("openrouter", "google/gemini-2.5-pro"),
    "grok-4": ("openrouter", "x-ai/grok-4"),
    "deepseek-r1": ("openrouter", "deepseek/deepseek-r1-0528"),
    # Google AI direct (LiteLLM gemini/* provider, GOOGLE_API_KEY).
    "gemini-3.1-pro-preview": ("gemini", "gemini-3.1-pro-preview"),
    "gemini-2.5-pro-direct": ("gemini", "gemini-2.5-pro"),
    "gemini-1.5-pro": ("gemini", "gemini-1.5-pro"),
    "gemini-2.5-flash-lite": ("gemini", "gemini-2.5-flash-lite"),
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-2.0-flash": ("gemini", "gemini-2.0-flash"),
}

PRESET_CHOICES: List[str] = sorted(PRESET_MAP.keys())


def model_default_decoding(provider: str, model: str) -> Dict[str, Optional[float]]:
    """Return model/provider default decoding settings."""

    lowered = model.lower()
    if provider == "local" and lowered.startswith("qwen3"):
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 32768,
        }
    if provider == "openai" and lowered.startswith("gpt-5"):
        return {
            "temperature": 1.0,  # Important: The following parameters are not supported when using GPT-5 models
            "top_p": None,  # Important: The following parameters are not supported when using GPT-5 models
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "claude" in lowered:
        return {
            "temperature": 1.0,  # Hard to find a recommended setting, using 1.0 as it was used for their AIME benchmark.
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "gemini" in lowered:
        return {
            "temperature": 1.0,  # default based on https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro
            "top_p": 0.95,  # default based on https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro althrough they refer to it as topP
            "max_tokens": 32768,
        }
    if provider == "gemini":
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "grok" in lowered:
        return {
            "temperature": 1.0,  # default based on https://docs.x.ai/docs/api-reference#chat-completions
            "top_p": 1.0,
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "deepseek" in lowered:
        return {
            "temperature": 1.0,  # based on https://api-docs.deepseek.com/quick_start/parameter_settings. It suggests various optoins but 1.0 for data analysis, 0.0 for coding/math.
            "top_p": 1.0,  # based on https://api-docs.deepseek.com/api/create-chat-completion
            "max_tokens": 32768,
        }
    return {}


# ---------------------------------------------------------------------------
# LM construction


def build_lm_from_preset(
    preset: str,
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    top_k: Optional[int],
    min_p: Optional[float],
    seed: Optional[int],
    port: int,
    configure: bool,
    num_retries: int = 8,
    cache: bool = False,
) -> Tuple[Any, str, Dict[str, Any]]:  # type: ignore[valid-type]
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")
    if preset not in PRESET_MAP:
        raise SystemExit(f"Unknown preset: {preset}")

    provider, model = PRESET_MAP[preset]

    defaults = model_default_decoding(provider, model)
    lm_kwargs: Dict[str, Any] = {k: v for k, v in defaults.items() if v is not None}
    if temperature is not None:
        lm_kwargs["temperature"] = temperature
    if top_p is not None:
        lm_kwargs["top_p"] = top_p
    if top_k is not None:
        lm_kwargs["top_k"] = top_k
    if min_p is not None:
        lm_kwargs["min_p"] = min_p
    if max_tokens is not None:
        lm_kwargs["max_tokens"] = max_tokens
    if seed is not None:
        lm_kwargs["seed"] = int(seed)

    if provider == "openai":
        if not _get_api_key("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (env/.env)")
        for unsupported in ("top_k", "min_p"):
            lm_kwargs.pop(unsupported, None)
        lm = dspy.LM(f"openai/{model}", num_retries=num_retries, cache=cache, **lm_kwargs)
        run_name = f"openai-{model}"
    elif provider == "local":
        api_base = f"http://localhost:{int(port)}/v1"
        lm = dspy.LM(
            model=f"openai/{model}",
            api_base=api_base,
            api_key="local",
            model_type="chat",
            num_retries=num_retries,
            cache=cache,
            **lm_kwargs,
        )
        run_name = f"local-{model.replace('/', '-')}"
    elif provider == "gemini":
        api_key = _get_api_key("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY is not set (env/.env)")
        for unsupported in ("top_k", "min_p", "seed"):
            lm_kwargs.pop(unsupported, None)
        lm = dspy.LM(
            model=f"gemini/{model}",
            api_key=api_key,
            num_retries=num_retries,
            cache=cache,
            **lm_kwargs,
        )
        run_name = f"gemini-{model.replace('/', '-')}"
    else:
        api_key = _get_api_key("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (env/.env)")
        lm = dspy.LM(
            model=f"openrouter/{model}",
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            num_retries=num_retries,
            cache=cache,
            **lm_kwargs,
        )
        run_name = f"openrouter-{model.replace('/', '-')}"

    if configure:
        dspy.configure(lm=lm, cache=False)

    lm_info = {
        "provider": provider,
        "model": model,
        "temperature": lm_kwargs.get("temperature"),
        "top_p": lm_kwargs.get("top_p"),
        "top_k": lm_kwargs.get("top_k"),
        "min_p": lm_kwargs.get("min_p"),
        "max_tokens": lm_kwargs.get("max_tokens"),
        "seed": lm_kwargs.get("seed"),
        "port": int(port) if provider == "local" else None,
        "num_retries": num_retries,
        "cache": cache,
    }
    return lm, run_name, lm_info


__all__ = [
    "PRESET_MAP",
    "PRESET_CHOICES",
    "model_default_decoding",
    "build_lm_from_preset",
]
