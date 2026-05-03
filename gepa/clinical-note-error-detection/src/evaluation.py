"""Evaluation helpers for the MEDIQA-CORR detection pipelines.

This module centralises the per-example evaluation, confusion-matrix plotting,
result serialisation, and the GEPA detection metric. It is intentionally
DSPy-aware (lazy import) but does not pull in model providers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from task_utils import (
    _lower,
    format_sentences_for_paper_prompt,
    humanize_error_type,
    normalize_error_type,
)

try:
    import dspy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - validated at runtime by callers
    dspy = None  # type: ignore


# ---------------------------------------------------------------------------
# Plot helpers


def _plot_binary_confusion(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    title: str,
    path: Path,
    tick_text: Optional[List[str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(f"Unable to plot confusion matrix (missing dependency): {exc}")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / row_totals
    annot = [
        [f"{count}\n{frac:.1%}" for count, frac in zip(row_counts, row_pct)]
        for row_counts, row_pct in zip(cm, pct)
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    display_labels = tick_text or [str(lab) for lab in labels]
    sns.heatmap(
        pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        linewidths=0.5,
        square=True,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=200)
        print(f"Saved confusion matrix → {path}")
    except Exception as exc:  # pragma: no cover - filesystem errors
        print(f"Failed to save confusion matrix ({exc})")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-example evaluation


def evaluate_programme(
    programme: Any,  # type: ignore[valid-type]
    examples: List["dspy.Example"],
) -> Dict[str, Any]:
    """Run ``programme`` over ``examples`` and compute MEDIQA-CORR detection metrics.

    Invokes the DSPy module once per example (``programme(sentences=ex.sentences)``)
    and compares the prediction against the gold ``verdict`` / ``error_sentence_id``
    / ``error_type`` fields. Note that this triggers one LM call per example and
    is therefore the main cost driver during baseline and post-GEPA evaluation.

    Args:
        programme: A DSPy module (typically ``PaperPromptProgramme``) whose
            ``forward`` accepts ``sentences: List[str]`` and returns a
            ``dspy.Prediction`` exposing ``verdict``, ``predicted_sentence_id``,
            ``corrected_sentence``, and ``raw_response``.
        examples: List of ``dspy.Example`` objects with the MEDIQA-CORR fields
            populated by ``dataset.to_examples`` (``sentences``, ``verdict``,
            ``error_sentence_id``, ``error_type``, ``reference_corrected_sentence``,
            ``reference_corrected_text``, ``text_id``).

    Returns:
        A dict aggregating all evaluation artefacts:

        * ``rows`` (List[Dict]): one row per example with raw + parsed fields,
          ground-truth alignment, and per-example correctness flags.
        * ``summary_df`` (pd.DataFrame): tabular view of ``rows``.
        * ``error_flag_accuracy`` (float): mean of ``pred_flag == gt_flag``.
        * ``error_sentence_accuracy`` (float): mean of
          ``pred_sentence_id == gt_sentence_id`` (lenient).
        * ``error_sentence_accuracy_strict`` (float): same, but requires
          verdict to also match (i.e. CORRECT cases must predict ``-1``).
        * ``binary_metrics`` (dict): accuracy, precision, recall, F1,
          specificity, FPR, balanced accuracy, MCC for the error-vs-correct
          binary task; entries are ``nan`` when the denominator is zero.
        * ``confusion_counts`` (dict): ``{"tp", "fp", "tn", "fn"}``.
        * ``detection_by_type_df`` / ``detection_by_type``: per-error-type
          flag-recall and sentence-recall breakdown (DataFrame + JSON-able dict).
        * ``y_true_flags`` / ``y_pred_flags`` (List[int]): aligned 0/1 vectors
          suitable for downstream sklearn-style scoring or plotting.

    Notes:
        - All metrics use the binary error-flag task (``"error"`` vs ``"correct"``);
          unparseable verdicts are coerced to ``"correct"``.
        - Predicted sentence IDs that fail to parse as ``int`` collapse to ``-1``.
    """

    rows: List[Dict[str, Any]] = []
    y_true_flags: List[int] = []
    y_pred_flags: List[int] = []

    for ex in examples:
        prediction = programme(sentences=ex.sentences)
        verdict = _lower(getattr(prediction, "verdict", ""))
        if verdict not in {"error", "correct"}:
            verdict = "correct"
        pred_flag = 1 if verdict == "error" else 0

        gt_flag = 1 if _lower(getattr(ex, "verdict", "correct")) == "error" else 0
        error_type = normalize_error_type(getattr(ex, "error_type", "none"))

        gt_sentence_id = int(getattr(ex, "error_sentence_id", -1))
        if gt_flag == 0:
            gt_sentence_id = -1

        raw_pred_sentence = getattr(prediction, "predicted_sentence_id", None)
        pred_sentence_id = -1
        if pred_flag == 1:
            try:
                pred_sentence_id = int(str(raw_pred_sentence).strip())
            except (TypeError, ValueError):
                pred_sentence_id = -1
        else:
            pred_sentence_id = -1

        sentence_detect_correct = 1.0 if pred_sentence_id == gt_sentence_id else 0.0
        sentence_detect_correct_strict = float(
            (gt_flag == 0 and pred_flag == 0 and pred_sentence_id == -1)
            or (gt_flag == 1 and pred_flag == 1 and pred_sentence_id == gt_sentence_id)
        )

        reference_corrected_sentence = getattr(ex, "reference_corrected_sentence", "NA")
        reference_corrected_text = getattr(ex, "reference_corrected_text", "")
        sentences_raw = "\n".join(ex.sentences)
        sentences_formatted = format_sentences_for_paper_prompt(ex.sentences)

        rows.append(
            {
                "text_id": getattr(ex, "text_id", ""),
                "pred_verdict": verdict,
                "pred_error_flag": pred_flag,
                "gt_error_flag": gt_flag,
                "error_type": error_type,
                "detect_correct": 1.0 if pred_flag == gt_flag else 0.0,
                "gt_sentence_id": gt_sentence_id,
                "pred_sentence_id": pred_sentence_id,
                "sentence_detect_correct": sentence_detect_correct,
                "sentence_detect_correct_strict": sentence_detect_correct_strict,
                "raw_response": getattr(prediction, "raw_response", None),
                "predicted_corrected_sentence": getattr(prediction, "corrected_sentence", None),
                "reference_corrected_sentence": reference_corrected_sentence,
                "reference_corrected_text": reference_corrected_text,
                "sentences_raw": sentences_raw,
                "sentences_formatted": sentences_formatted,
            }
        )

        y_true_flags.append(gt_flag)
        y_pred_flags.append(pred_flag)

    summary_df = pd.DataFrame(rows)

    error_flag_accuracy = float("nan")
    error_sentence_accuracy = float("nan")
    error_sentence_accuracy_strict = float("nan")
    detection_by_type_df = pd.DataFrame()
    detection_by_type_json: Dict[str, Dict[str, Any]] = {}

    if not summary_df.empty:
        error_flag_accuracy = float(summary_df["detect_correct"].mean())
        error_sentence_accuracy = float(summary_df["sentence_detect_correct"].mean())
        error_sentence_accuracy_strict = float(summary_df["sentence_detect_correct_strict"].mean())

        err_only = summary_df[(summary_df["gt_error_flag"] == 1) & (summary_df["error_type"] != "none")]
        if not err_only.empty:
            flag_recall_series = err_only.groupby("error_type")["pred_error_flag"].mean().rename("flag_recall")
            sentence_recall_series = (
                err_only.groupby("error_type")["sentence_detect_correct"].mean().rename("sentence_recall")
            )
            count_series = err_only.groupby("error_type")[["sentence_detect_correct"]].count().rename(
                columns={"sentence_detect_correct": "count"}
            )
            detection_by_type_df = pd.concat(
                [flag_recall_series, sentence_recall_series, count_series], axis=1
            ).sort_index()
            detection_by_type_df["flag_recall"] = detection_by_type_df["flag_recall"].astype(float)
            detection_by_type_df["sentence_recall"] = detection_by_type_df["sentence_recall"].astype(float)
            detection_by_type_df["count"] = detection_by_type_df["count"].astype(int)
            detection_by_type_df.index.name = "error_type"
            detection_by_type_json = {
                err: {
                    "label": humanize_error_type(err),
                    "flag_recall": float(row["flag_recall"]),
                    "sentence_recall": float(row["sentence_recall"]),
                    "count": int(row["count"]),
                }
                for err, row in detection_by_type_df.iterrows()
            }

    tp = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 1 and yp == 0)

    def _safe_div(num: float, den: float) -> float:
        return num / den if den else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)

    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if math.isnan(recall) or math.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = (recall + specificity) / 2

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_den:
        mcc = ((tp * tn) - (fp * fn)) / mcc_den
    else:
        mcc = float("nan")

    binary_metrics = {
        "accuracy": error_flag_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }

    confusion_counts = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    return {
        "rows": rows,
        "summary_df": summary_df,
        "error_flag_accuracy": error_flag_accuracy,
        "error_sentence_accuracy": error_sentence_accuracy,
        "error_sentence_accuracy_strict": error_sentence_accuracy_strict,
        "binary_metrics": binary_metrics,
        "confusion_counts": confusion_counts,
        "detection_by_type_df": detection_by_type_df,
        "detection_by_type": detection_by_type_json,
        "y_true_flags": y_true_flags,
        "y_pred_flags": y_pred_flags,
    }


# ---------------------------------------------------------------------------
# Result serialisation


def _clean_value(val: Any) -> Any:
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def save_result_outputs(
    run_dir: Path,
    split_name: str,
    variant: str,
    result: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    prefix = f"{variant}_{split_name}"
    predictions_path = run_dir / f"{prefix}_predictions.csv"
    pd.DataFrame(result["rows"]).to_csv(predictions_path, index=False)

    metrics_path = run_dir / f"{prefix}_metrics_per_example.csv"
    result["summary_df"].to_csv(metrics_path, index=False)

    confusion_png_path = run_dir / f"{prefix}_confusion.png"
    confusion_pdf_path = run_dir / f"{prefix}_confusion.pdf"
    y_true = result.get("y_true_flags", [])
    y_pred = result.get("y_pred_flags", [])
    if y_true and y_pred:
        title = f"{variant} {split_name}"
        _plot_binary_confusion(
            y_true,
            y_pred,
            [0, 1],
            title,
            confusion_png_path,
            tick_text=["0 = correct", "1 = error"],
        )
        _plot_binary_confusion(
            y_true,
            y_pred,
            [0, 1],
            title,
            confusion_pdf_path,
            tick_text=["0 = correct", "1 = error"],
        )
        confusion_png_str = str(confusion_png_path)
        confusion_pdf_str = str(confusion_pdf_path)
    else:
        confusion_png_str = None
        confusion_pdf_str = None

    overall_metrics_payload = {
        "error_flag_accuracy": _clean_value(result.get("error_flag_accuracy")),
        "error_sentence_accuracy": _clean_value(result.get("error_sentence_accuracy")),
        "error_sentence_accuracy_strict": _clean_value(result.get("error_sentence_accuracy_strict")),
        "binary_metrics": {
            key: _clean_value(val)
            for key, val in (result.get("binary_metrics") or {}).items()
        },
        "confusion_counts": result.get("confusion_counts"),
        "totals": {
            "n": int(len(result.get("y_true_flags", []))),
            "n_errors": int(sum(result.get("y_true_flags", []))),
            "n_correct": int(len(result.get("y_true_flags", [])) - sum(result.get("y_true_flags", []))),
        },
    }
    metrics_json_path = run_dir / f"{prefix}_metrics_overall.json"
    metrics_json_path.write_text(json.dumps(overall_metrics_payload, indent=2) + "\n", encoding="utf-8")

    by_type_df = result.get("detection_by_type_df")
    by_type_csv_path: Optional[Path] = None
    by_type_json_path: Optional[Path] = None
    if isinstance(by_type_df, pd.DataFrame) and not by_type_df.empty:
        by_type_csv_path = run_dir / f"{prefix}_metrics_by_error_type.csv"
        by_type_df.to_csv(by_type_csv_path, index=True)
        by_type_json_path = run_dir / f"{prefix}_metrics_by_error_type.json"
        by_type_json_path.write_text(
            json.dumps(result.get("detection_by_type", {}), indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "metrics_json": str(metrics_json_path),
        "metrics_by_type_csv": str(by_type_csv_path) if by_type_csv_path else None,
        "metrics_by_type_json": str(by_type_json_path) if by_type_json_path else None,
        "confusion_png": confusion_png_str,
        "confusion_pdf": confusion_pdf_str,
    }


# ---------------------------------------------------------------------------
# GEPA metric (detection only)


def gepa_detection_metric(gt, pred, trace=None, pred_name=None, pred_trace=None):  # type: ignore[no-untyped-def]
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")

    gt_verdict = _lower(getattr(gt, "verdict", ""))
    pred_verdict = _lower(getattr(pred, "verdict", ""))
    err_type_code = normalize_error_type(getattr(gt, "error_type", "none"))
    err_label = humanize_error_type(err_type_code)
    err_sentence = (getattr(gt, "error_sentence", "") or "").strip()
    ref_sentence = (getattr(gt, "reference_corrected_sentence", "") or "").strip()
    ref_text = (getattr(gt, "reference_corrected_text", "") or "").strip()

    if pred_verdict not in {"error", "correct"}:
        fb = "Format error: respond with 'CORRECT' or '<sent_id> <correction>'."
        return dspy.Prediction(score=0.0, feedback=fb)

    detail_bits: List[str] = []
    if err_type_code != "none":
        detail_bits.append(f"Error type: {err_label}.")
    else:
        detail_bits.append("No medical error in this case.")
    if err_sentence and err_sentence.upper() != "NA":
        detail_bits.append(f"Erroneous sentence: \"{err_sentence}\".")
    if ref_sentence and ref_sentence.upper() != "NA":
        detail_bits.append(f"Reference correction: \"{ref_sentence}\".")
    if ref_text and ref_text.upper() != "NA":
        detail_bits.append(f"Corrected text: \"{ref_text}\".")
    detail_str = " ".join(detail_bits).strip()

    if pred_verdict == gt_verdict:
        prefix = "True positive" if gt_verdict == "error" else "True negative"
        fb = f"{prefix}: correct prediction. {detail_str}".strip()
        return dspy.Prediction(score=1.0, feedback=fb)

    prefix = "False negative" if gt_verdict == "error" else "False positive"
    fb = (
        f"{prefix}: predicted {pred_verdict.upper()} while true label is {gt_verdict.upper()}. "
        f"{detail_str}"
    ).strip()
    return dspy.Prediction(score=0.0, feedback=fb)


__all__ = [
    "_plot_binary_confusion",
    "_clean_value",
    "evaluate_programme",
    "save_result_outputs",
    "gepa_detection_metric",
]
