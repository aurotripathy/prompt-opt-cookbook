"""Dataset loading and DSPy example conversion for MEDIQA-CORR.

Centralises the CSV → DataFrame → ``dspy.Example`` pipeline shared by
``detect_gepa.py`` and ``detect_eval.py``. Pure schema/IO concerns; no model
or evaluation logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import pandas as pd

from task_utils import _lower, normalize_error_type, split_sentences_blob

try:
    import dspy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - validated at runtime by callers
    dspy = None  # type: ignore


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a MEDEC CSV, dropping rows that are entirely blank.

    The ``dropna(how="all")`` guards against trailing empty rows that some MEDEC
    CSV exports carry. Callers may layer additional transforms (e.g. shuffle,
    head) on the returned frame.
    """
    return pd.read_csv(path).dropna(how="all")


def to_examples(df: pd.DataFrame) -> List["dspy.Example"]:
    """Convert a MEDEC DataFrame into a list of ``dspy.Example``.

    The output ``Example`` carries every label field both pipelines might want:
    ``reference_corrected_sentence`` (used by all consumers) and
    ``reference_corrected_text`` (used by ``evaluation.evaluate_programme`` /
    ``gepa_detection_metric``). Consumers that don't need a field simply ignore
    it via ``getattr(..., default)``.
    """
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")

    examples: List[dspy.Example] = []
    for _, row in df.iterrows():
        sentences = split_sentences_blob(row.get("Sentences", ""))
        flag = int(row.get("Error Flag", 0) or 0)
        verdict = "error" if flag == 1 else "correct"
        err_type = normalize_error_type(row.get("Error Type", "none"))
        if verdict == "correct":
            err_type = "none"

        raw_sentence_id = row.get("Error Sentence ID", -1)
        try:
            err_sentence_id = int(raw_sentence_id)
        except (TypeError, ValueError):
            err_sentence_id = -1
        if verdict == "correct":
            err_sentence_id = -1

        reference_correction = str(row.get("Corrected Sentence", "NA"))
        reference_corrected_text = str(row.get("Corrected Text", ""))

        example = dspy.Example(
            text_id=str(row.get("Text ID", "")),
            sentences=sentences,
            verdict=verdict,
            error_type=err_type,
            error_sentence=str(row.get("Error Sentence", "NA")),
            error_sentence_id=err_sentence_id,
            reference_corrected_sentence=reference_correction,
            reference_corrected_text=reference_corrected_text,
        ).with_inputs("sentences")
        examples.append(example)
    return examples


def stratified_first_k(examples: List["dspy.Example"], k: int) -> List["dspy.Example"]:
    """Take the first ``k`` examples while keeping the error/correct ratio.

    Used by ``detect_gepa.py`` to cap the GEPA train and pareto-val splits
    without losing class balance. ``k <= 0`` or ``k >= len(examples)`` returns
    the input unchanged.
    """
    if not k or k <= 0 or k >= len(examples):
        return examples
    pos = [e for e in examples if _lower(getattr(e, "verdict", "correct")) == "error"]
    neg = [e for e in examples if _lower(getattr(e, "verdict", "correct")) != "error"]
    total = max(len(examples), 1)
    k_pos = int(round(k * (len(pos) / total)))
    k_pos = min(k_pos, len(pos))
    k_neg = min(k - k_pos, len(neg))
    selected = pos[:k_pos] + neg[:k_neg]
    if len(selected) < k:
        remainder = pos[k_pos:] + neg[k_neg:]
        selected += remainder[: (k - len(selected))]
    return selected


__all__ = ["load_csv", "to_examples", "stratified_first_k"]
