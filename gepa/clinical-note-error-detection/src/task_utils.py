"""Shared utilities for `detect_gepa.py` and `detect_eval.py`.

Covers MEDIQA-CORR sentence parsing, error-type normalisation, the paper-prompt
input/output format, and reproducibility helpers (RNG seeding). Intentionally
free of any DSPy / model dependencies so it can be imported from anywhere.
"""

from __future__ import annotations

import os
import random
import secrets
import string
from typing import Any, List, Optional, Tuple


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (silently skipping unavailable libs)."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def short_id(n: int = 5) -> str:
    """Return a short lowercase-alphanumeric ID of length `n` (>=1)."""

    alphabet = string.ascii_lowercase + string.digits
    length = max(int(n), 1)
    return "".join(secrets.choice(alphabet) for _ in range(length))


def split_sentences_blob(blob: str) -> List[str]:
    if not isinstance(blob, str):
        return []
    return [ln.strip("\r") for ln in blob.split("\n") if ln.strip() != ""]


def normalize_error_type(raw: Any) -> str:
    if raw is None:
        return "none"
    text = str(raw).strip().lower()
    if text == "":
        return "none"
    mapping = {
        "none": "none",
        "no error": "none",
        "diagnosis": "diagnosis",
        "management": "management",
        "treatment": "treatment",
        "pharmacotherapy": "pharmacotherapy",
        "pharmacology": "pharmacotherapy",
        "causal organism": "causalorganism",
        "causalorganism": "causalorganism",
        "causal-organism": "causalorganism",
    }
    return mapping.get(text, text)


def humanize_error_type(code: str) -> str:
    mapping = {
        "none": "none",
        "diagnosis": "diagnosis",
        "management": "management",
        "treatment": "treatment",
        "pharmacotherapy": "pharmacotherapy",
        "causalorganism": "causal organism",
    }
    return mapping.get(code, code)


def _lower(x: Any) -> str:
    return (str(x) if x is not None else "").strip().lower()


def _split_sentence_idx_and_text(raw: str, fallback_idx: int) -> Tuple[str, str]:
    stripped = (raw or "").strip()
    if stripped == "":
        return str(fallback_idx), ""

    if "|" in stripped:
        left, right = stripped.split("|", 1)
        idx = left.strip() or str(fallback_idx)
        text = right.strip(" :-")
        return idx, text.strip()

    parts = stripped.split(" ", 1)
    if parts and parts[0].isdigit():
        idx = parts[0]
        text = parts[1].strip() if len(parts) > 1 else ""
        return idx, text

    return str(fallback_idx), stripped


def format_sentences_for_paper_prompt(sentences: List[str]) -> str:
    """Render sentences in the MEDEC paper-prompt format: ``"<id>| <text>"`` per line.

    Each input string is parsed with :func:`_split_sentence_idx_and_text` so that
    pre-existing IDs (``"3| ..."`` or ``"3 ..."``) are preserved; otherwise the
    enumeration index is used as the fallback ID. The result is a single newline-
    joined block ready to drop into the LM prompt as the ``input`` field.

    Example::

        >>> format_sentences_for_paper_prompt(["Patient is stable.", "BP normal."])
        '0| Patient is stable.\\n1| BP normal.'
    """

    formatted: List[str] = []
    for idx, raw in enumerate(sentences):
        sent_idx, sent_text = _split_sentence_idx_and_text(raw, idx)
        formatted.append(f"{sent_idx}| {sent_text}")
    return "\n".join(formatted)


def parse_paper_prompt_output(raw: str) -> Tuple[str, Optional[int], str]:
    text = (raw or "").strip()
    if text == "":
        return "correct", None, ""

    upper = text.upper()
    if upper.startswith("CORRECT"):
        return "correct", None, text

    first_token, remainder = text, ""
    if " " in text:
        first_token, remainder = text.split(" ", 1)
    token_clean = first_token.rstrip(":").rstrip("-")

    if token_clean.isdigit():
        try:
            sent_id = int(token_clean)
        except ValueError:
            sent_id = None
        return "error", sent_id, remainder.strip()

    return "error", None, text


__all__ = [
    "set_seed",
    "short_id",
    "split_sentences_blob",
    "normalize_error_type",
    "humanize_error_type",
    "_lower",
    "_split_sentence_idx_and_text",
    "format_sentences_for_paper_prompt",
    "parse_paper_prompt_output",
]
