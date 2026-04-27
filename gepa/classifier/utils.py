"""Shared helpers for the GEPA classifier notebook / script.

Provides Gemini model resolution: given a preference list of LiteLLM-style
Gemini model ids (e.g. "gemini/gemini-2.5-pro"), probe each one with a trivial
call and return the first that actually responds to the current
``GOOGLE_API_KEY``. This sidesteps 404s for preview / allow-listed models.
"""

from __future__ import annotations

import os
from typing import Sequence


SMALL_MODEL_CANDIDATES: list[str] = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash",
]

LARGE_MODEL_CANDIDATES: list[str] = [
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-1.5-pro",
]


def resolve_gemini_model(candidates: Sequence[str], role: str) -> str:
    """Return the first model id in ``candidates`` that responds to a trivial prompt.

    Each candidate is probed with a tiny ``dspy.LM(...)`` call. Failures (404,
    401, quota, etc.) are logged and the next candidate is tried. Raises
    ``RuntimeError`` if none work or if ``GOOGLE_API_KEY`` is missing.

    Args:
        candidates: Ordered preference list of LiteLLM Gemini model ids,
            e.g. ``["gemini/gemini-2.5-pro", "gemini/gemini-1.5-pro"]``.
        role: Short label used in probe output (e.g. ``"small_model"``) so logs
            make it clear which slot is being resolved.

    Returns:
        The first usable model id from ``candidates``.
    """
    import dspy

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Put it in .env at the repo root or export it before running."
        )

    last_err: Exception | None = None
    for model_id in candidates:
        try:
            lm = dspy.LM(model_id, api_key=api_key, cache=False, max_tokens=8)
            lm("ping")
            print(f"[model-probe] {role}: using {model_id}")
            return model_id
        except Exception as err:
            msg = str(err).splitlines()[0][:180]
            print(f"[model-probe] {role}: {model_id} unavailable ({msg})")
            last_err = err

    raise RuntimeError(
        f"No usable Gemini model found for role={role!r} among {list(candidates)}. "
        f"Last error: {last_err}"
    )


__all__ = [
    "SMALL_MODEL_CANDIDATES",
    "LARGE_MODEL_CANDIDATES",
    "resolve_gemini_model",
]
