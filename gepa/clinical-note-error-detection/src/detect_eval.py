"""Single-pass MEDIQA-CORR detection benchmark without GEPA optimisation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dataset import load_csv, to_examples
from evaluation import _clean_value, _plot_binary_confusion
from models import PRESET_CHOICES, build_lm_from_preset
from task_utils import (
    _lower,
    _split_sentence_idx_and_text,
    format_sentences_for_paper_prompt,
    humanize_error_type,
    normalize_error_type,
    parse_paper_prompt_output,
    set_seed,
    short_id,
    split_sentences_blob,
)


PAPER_PROMPT_TEXT = (
    "The following is a medical narrative about a patient. You are a skilled medical "
    "doctor reviewing the clinical text. The text is either correct or contains one "
    "error. The text has one sentence per line. Each line starts with the sentence ID, "
    "followed by a pipe character then the sentence to check. Check every sentence of "
    "the text. If the text is correct return the following output: CORRECT. If the text "
    "has a medical error related to treatment, management, cause, or diagnosis, return "
    "the sentence id of the sentence containing the error, followed by a space, and "
    "then a corrected version of the sentence. Finding and correcting the error "
    "requires medical knowledge and reasoning."
)


try:
    import dspy  # type: ignore
    from typing import Literal  # noqa: WPS300 (re-export for typing)

    class DetectError(dspy.Signature):
        """Simple detection baseline."""

        sentences: List[str] = dspy.InputField()
        verdict: Literal["error", "correct"] = dspy.OutputField()

    try:
        DetectError.model_rebuild()
    except Exception:
        pass

    class DetectProgramme(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detect = dspy.Predict(DetectError)

        def forward(self, sentences: List[str]) -> dspy.Prediction:
            raw_pred = self.detect(sentences=sentences)
            verdict = _lower(getattr(raw_pred, "verdict", ""))
            if verdict not in {"error", "correct"}:
                verdict = "correct"
            return dspy.Prediction(verdict=verdict)

    class PaperPromptSignature(dspy.Signature):
        __doc__ = PAPER_PROMPT_TEXT

        input: str = dspy.InputField()
        output: str = dspy.OutputField()

    try:
        PaperPromptSignature.model_rebuild()
    except Exception:
        pass

    class PaperPromptProgramme(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detect = dspy.Predict(PaperPromptSignature)

        def forward(self, sentences: List[str]) -> dspy.Prediction:
            formatted = format_sentences_for_paper_prompt(sentences)
            completion = self.detect(input=formatted)
            raw_response = str(getattr(completion, "output", "")).strip()

            verdict, parsed_sentence_id, parsed_correction = parse_paper_prompt_output(raw_response)
            normalized_verdict = "error" if verdict.lower() == "error" else "correct"

            sentence_id_str: Optional[str] = None
            if parsed_sentence_id is not None:
                sentence_id_str = str(parsed_sentence_id)

            return dspy.Prediction(
                verdict=normalized_verdict,
                raw_response=raw_response,
                predicted_sentence_id=sentence_id_str,
                corrected_sentence=parsed_correction,
                error_category=None,
            )

except ModuleNotFoundError as exc:  # pragma: no cover - validated at runtime
    dspy = None  # type: ignore
    Literal = None  # type: ignore  # noqa: N816 (maintain compatibility)
    DETECT_IMPORT_ERROR = exc
else:
    DETECT_IMPORT_ERROR = None


def evaluate_programme(
    programme: Any,  # type: ignore[valid-type]
    examples: List["dspy.Example"],
) -> Dict[str, Any]:
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
                "predicted_error_category": getattr(prediction, "error_category", None),
                "predicted_corrected_sentence": getattr(prediction, "corrected_sentence", None),
                "reference_corrected_sentence": reference_corrected_sentence,
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


def save_split_outputs(
    run_dir: Path,
    split_name: str,
    run_label: str,
    result: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    split_dir = run_dir
    predictions_path = split_dir / f"predictions_{split_name}.csv"
    pd.DataFrame(result["rows"]).to_csv(predictions_path, index=False)

    metrics_path = split_dir / f"metrics_per_example_{split_name}.csv"
    result["summary_df"].to_csv(metrics_path, index=False)

    confusion_path = split_dir / f"{split_name}_confusion_matrix.png"
    y_true = result.get("y_true_flags", [])
    y_pred = result.get("y_pred_flags", [])
    if y_true and y_pred:
        _plot_binary_confusion(
            y_true,
            y_pred,
            [0, 1],
            f"{run_label}\n{split_name}",
            confusion_path,
            tick_text=["0 = correct", "1 = error"],
        )
        confusion_str = str(confusion_path)
    else:
        confusion_str = None

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
    metrics_json_path = split_dir / f"metrics_overall_{split_name}.json"
    metrics_json_path.write_text(json.dumps(overall_metrics_payload, indent=2) + "\n", encoding="utf-8")

    by_type_df = result.get("detection_by_type_df")
    by_type_csv_path: Optional[Path] = None
    by_type_json_path: Optional[Path] = None
    if isinstance(by_type_df, pd.DataFrame) and not by_type_df.empty:
        by_type_csv_path = split_dir / f"metrics_by_error_type_{split_name}.csv"
        by_type_df.to_csv(by_type_csv_path, index=True)
        by_type_json_path = split_dir / f"metrics_by_error_type_{split_name}.json"
        by_type_json_path.write_text(
            json.dumps(result.get("detection_by_type", {}), indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "confusion_png": confusion_str,
        "metrics_json": str(metrics_json_path),
        "metrics_by_type_csv": str(by_type_csv_path) if by_type_csv_path else None,
        "metrics_by_type_json": str(by_type_json_path) if by_type_json_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-csv",
        "--val-csv",
        dest="val_csv",
        metavar="EVAL_CSV",
        default="data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
        help=(
            "CSV to evaluate against (any split: val, test, etc.). "
            "'--val-csv' is accepted as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--limit-eval",
        "--limit-val",
        dest="limit_val",
        metavar="LIMIT_EVAL",
        type=int,
        default=0,
        help="Cap rows loaded from the eval CSV (0 = all). '--limit-val' alias retained.",
    )
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="qwen3-8b")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent repeats.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, dest="top_p", default=None)
    parser.add_argument("--top-k", type=int, dest="top_k", default=None)
    parser.add_argument("--min-p", type=float, dest="min_p", default=None)
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", default=None)
    parser.add_argument("--port", type=int, default=7501)
    parser.add_argument(
        "--prompt",
        choices=["paper", "detect"],
        default="paper",
        help="Inference prompt style (paper prompt matches the published baseline).",
    )
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="medec-detect-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    if DETECT_IMPORT_ERROR is not None:
        raise SystemExit(
            "DSPy is required for this script but could not be imported: "
            f"{DETECT_IMPORT_ERROR}",
        )

    set_seed(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df_val = load_csv(args.val_csv)

    try:
        if (df_val["Error Flag"] == 1).any():
            min_id = pd.to_numeric(
                df_val.loc[df_val["Error Flag"] == 1, "Error Sentence ID"], errors="coerce"
            ).min()
            if pd.notna(min_id) and int(min_id) < 0:
                print("Warning: Negative Error Sentence ID found in validation CSV.")
    except Exception:
        pass

    if args.limit_val and args.limit_val > 0:
        df_val = df_val.head(args.limit_val)

    valset = to_examples(df_val)

    lm, run_name, lm_info = build_lm_from_preset(
        args.preset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        port=args.port,
        configure=True,
    )

    ts = dt.datetime.now().strftime("%Y_%m_%d-%H%M")
    suffix = short_id()
    run_dir = out_root / f"detect_eval_{run_name}_{args.prompt}_{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_module = None
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore

            wandb_module = wandb
            wandb_run_name = args.wandb_run_name or f"{run_name}-{args.prompt}-{ts}-{suffix}"
            wandb_config = {
                "preset": args.preset,
                "prompt": args.prompt,
                "runs": int(args.runs),
                "seed": int(args.seed),
                "seeds": [int(s) for s in range(args.seed, args.seed + args.runs)],
                "lm": lm_info,
                "val_csv": str(args.val_csv),
                "limit_val": int(args.limit_val),
            }
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
            )
        except Exception as exc:
            print(f"W&B disabled: {exc}")
            wandb_module = None
            wandb_run = None

    print("\nRun configuration")
    print(f"  Preset: {args.preset}")
    print(f"  Provider: {lm_info['provider']}")
    print(f"  Model: {lm_info['model']}")
    print(f"  Temperature: {lm_info['temperature']}")
    print(
        "  TopP: {top_p}  TopK: {top_k}  MinP: {min_p}".format(
            top_p=lm_info.get("top_p"),
            top_k=lm_info.get("top_k"),
            min_p=lm_info.get("min_p"),
        )
    )
    print(f"  Max tokens: {lm_info['max_tokens']}")
    print(f"  Local port: {lm_info['port']}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Seed: {args.seed}")
    print(f"  Val examples: {len(valset)}")
    print(f"  Output directory: {run_dir}\n")

    seeds = [args.seed + i for i in range(args.runs)]
    per_run: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for run_idx, seed_val in enumerate(seeds, 1):
        print(f"=== Repeat {run_idx}/{args.runs} seed={seed_val} ===")
        set_seed(seed_val)
        build_lm_from_preset(
            args.preset,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=seed_val,
            port=args.port,
            configure=True,
        )
        programme = PaperPromptProgramme() if args.prompt == "paper" else DetectProgramme()
        val_result = evaluate_programme(programme, valset)
        acc = float(val_result.get("error_flag_accuracy", float("nan")))

        if math.isnan(acc):
            print("  Error flag accuracy: nan")
        else:
            print(f"  Error flag accuracy: {acc:.3f}")

        run_dir_i = run_dir / f"rep{run_idx}"
        run_dir_i.mkdir(parents=True, exist_ok=True)
        paths_val = save_split_outputs(
            run_dir_i,
            "val",
            f"{run_name} ({args.prompt}) rep{run_idx}",
            val_result,
        )

        detection_json = val_result.get("detection_by_type", {})
        if detection_json:
            print("  Recall by error type:")
            for err_code, stats_dict in detection_json.items():
                label = humanize_error_type(err_code)
                flag_recall_val = stats_dict.get("flag_recall")
                sentence_recall_val = stats_dict.get("sentence_recall")
                count = stats_dict.get("count")
                flag_str = f"{flag_recall_val:.3f}" if flag_recall_val is not None else "nan"
                sent_str = f"{sentence_recall_val:.3f}" if sentence_recall_val is not None else "nan"
                print(
                    f"    {label}: flag_recall={flag_str} sentence_recall={sent_str} (n={count})"
                )

        run_binary_metrics = val_result.get("binary_metrics", {})
        run_confusion = val_result.get("confusion_counts", {})
        error_sentence_accuracy = val_result.get("error_sentence_accuracy")

        if wandb_run is not None and wandb_module is not None:
            log_payload = {
                "run/index": run_idx,
                "run/seed": seed_val,
            }
            if not math.isnan(acc):
                log_payload["val/error_flag_accuracy"] = acc
            if error_sentence_accuracy is not None and not math.isnan(error_sentence_accuracy):
                log_payload["val/error_sentence_accuracy"] = error_sentence_accuracy
            esa_strict = val_result.get("error_sentence_accuracy_strict")
            if esa_strict is not None and not math.isnan(esa_strict):
                log_payload["val/error_sentence_accuracy_strict"] = esa_strict
            for metric_name in [
                "precision",
                "recall",
                "f1",
                "specificity",
                "fpr",
                "balanced_accuracy",
                "mcc",
            ]:
                metric_value = run_binary_metrics.get(metric_name)
                if metric_value is not None and not math.isnan(metric_value):
                    log_payload[f"val/{metric_name}"] = metric_value
            wandb_module.log(log_payload)
            if paths_val["confusion_png"]:
                try:
                    wandb_module.log(
                        {
                            f"val/confusion_rep{run_idx}": wandb_module.Image(paths_val["confusion_png"]),
                        }
                    )
                except Exception:
                    pass

        per_run.append(
            {
                "run": run_idx,
                "seed": seed_val,
                "error_flag_accuracy": None if math.isnan(acc) else acc,
                "error_sentence_accuracy": _clean_value(error_sentence_accuracy),
                "error_sentence_accuracy_strict": _clean_value(val_result.get("error_sentence_accuracy_strict")),
                "precision": _clean_value(run_binary_metrics.get("precision")),
                "recall": _clean_value(run_binary_metrics.get("recall")),
                "f1": _clean_value(run_binary_metrics.get("f1")),
                "specificity": _clean_value(run_binary_metrics.get("specificity")),
                "fpr": _clean_value(run_binary_metrics.get("fpr")),
                "balanced_accuracy": _clean_value(run_binary_metrics.get("balanced_accuracy")),
                "mcc": _clean_value(run_binary_metrics.get("mcc")),
                "confusion_counts": run_confusion,
                "predictions_csv": paths_val["predictions_csv"],
                "metrics_csv": paths_val["metrics_csv"],
                "confusion_png": paths_val["confusion_png"],
                "metrics_json": paths_val.get("metrics_json"),
                "metrics_by_type_csv": paths_val.get("metrics_by_type_csv"),
                "metrics_by_type_json": paths_val.get("metrics_by_type_json"),
                "detection_by_error_type": detection_json,
            }
        )

        if args.debug:
            debug_rows.append(
                {
                    "run": run_idx,
                    "seed": seed_val,
                    "rows": val_result.get("rows", [])[:5],
                }
            )

    metric_field_map = {
        "error_flag_accuracy": "val/error_flag_accuracy",
        "error_sentence_accuracy": "val/error_sentence_accuracy",
        "error_sentence_accuracy_strict": "val/error_sentence_accuracy_strict",
        "precision": "val/precision",
        "recall": "val/recall",
        "f1": "val/f1",
        "specificity": "val/specificity",
        "fpr": "val/fpr",
        "balanced_accuracy": "val/balanced_accuracy",
        "mcc": "val/mcc",
    }

    aggregate_metrics: Dict[str, Dict[str, Any]] = {}
    for field, summary_key in metric_field_map.items():
        values = []
        for entry in per_run:
            val = entry.get(field)
            if val is not None and not math.isnan(val):
                values.append(val)
        if values:
            mean_val = float(stats.mean(values))
            std_val = float(stats.pstdev(values)) if len(values) > 1 else 0.0
        else:
            mean_val = float("nan")
            std_val = float("nan")
        aggregate_metrics[summary_key] = {
            "mean": _clean_value(mean_val),
            "std": _clean_value(std_val),
        }

    mean_acc = aggregate_metrics["val/error_flag_accuracy"]["mean"]
    std_acc = aggregate_metrics["val/error_flag_accuracy"]["std"]

    aggregate_by_type_temp: Dict[str, Dict[str, Any]] = {}
    for entry in per_run:
        per_type = entry.get("detection_by_error_type") or {}
        for err_code, stats_dict in per_type.items():
            bucket = aggregate_by_type_temp.setdefault(
                err_code,
                {
                    "flag": [],
                    "sentence": [],
                    "count": stats_dict.get("count"),
                    "label": stats_dict.get("label"),
                },
            )
            if stats_dict.get("label"):
                bucket["label"] = stats_dict.get("label")
            flag_val = stats_dict.get("flag_recall")
            sentence_val = stats_dict.get("sentence_recall")
            if flag_val is not None and not math.isnan(flag_val):
                bucket["flag"].append(flag_val)
            if sentence_val is not None and not math.isnan(sentence_val):
                bucket["sentence"].append(sentence_val)

    aggregate_by_type: Dict[str, Dict[str, Any]] = {}

    def _mean_std(values: List[float]) -> Tuple[Any, Any]:
        if not values:
            return _clean_value(float("nan")), _clean_value(float("nan"))
        if len(values) == 1:
            return _clean_value(float(values[0])), _clean_value(0.0)
        return _clean_value(float(stats.mean(values))), _clean_value(float(stats.pstdev(values)))

    for err_code, bucket in aggregate_by_type_temp.items():
        flag_mean, flag_std = _mean_std(bucket["flag"])
        sentence_mean, sentence_std = _mean_std(bucket["sentence"])
        aggregate_by_type[err_code] = {
            "label": bucket.get("label") or humanize_error_type(err_code),
            "flag_recall_mean": flag_mean,
            "flag_recall_std": flag_std,
            "sentence_recall_mean": sentence_mean,
            "sentence_recall_std": sentence_std,
            "count": bucket.get("count"),
        }

    summary_payload: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "preset": args.preset,
        "prompt": args.prompt,
        "seed": int(args.seed),
        "val_csv": str(args.val_csv),
        "lm_info": lm_info,
        "per_run": per_run,
        "aggregate": {
            "runs": len(per_run),
            "seeds": seeds,
            "val/error_flag_accuracy_mean": mean_acc,
            "val/error_flag_accuracy_std": std_acc,
            "val/error_sentence_accuracy_mean": aggregate_metrics["val/error_sentence_accuracy"]["mean"],
            "val/error_sentence_accuracy_std": aggregate_metrics["val/error_sentence_accuracy"]["std"],
            "val/error_sentence_accuracy_strict_mean": aggregate_metrics["val/error_sentence_accuracy_strict"]["mean"],
            "val/error_sentence_accuracy_strict_std": aggregate_metrics["val/error_sentence_accuracy_strict"]["std"],
            "n_val_examples": int(len(valset)),
            "metrics": aggregate_metrics,
            "by_error_type": aggregate_by_type,
        },
        "splits": {
            "val": {
                "n_examples": int(len(valset)),
                "error_flag_accuracy_mean": mean_acc,
                "error_flag_accuracy_std": std_acc,
                "error_sentence_accuracy_mean": aggregate_metrics["val/error_sentence_accuracy"]["mean"],
                "error_sentence_accuracy_std": aggregate_metrics["val/error_sentence_accuracy"]["std"],
                "error_sentence_accuracy_strict_mean": aggregate_metrics["val/error_sentence_accuracy_strict"]["mean"],
                "error_sentence_accuracy_strict_std": aggregate_metrics["val/error_sentence_accuracy_strict"]["std"],
            }
        },
        "wandb_project": args.wandb_project if wandb_run is not None else None,
        "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    def _fmt_value(val: Any) -> str:
        if isinstance(val, float) and not math.isnan(val):
            return f"{val:.4f}"
        return "nan"

    def _fmt_pair(mean_val: Any, std_val: Any) -> str:
        return f"{_fmt_value(mean_val)} ± {_fmt_value(std_val)}"

    ef_mean = summary_payload["aggregate"].get("val/error_flag_accuracy_mean")
    ef_std = summary_payload["aggregate"].get("val/error_flag_accuracy_std")
    es_mean = summary_payload["aggregate"].get("val/error_sentence_accuracy_mean")
    es_std = summary_payload["aggregate"].get("val/error_sentence_accuracy_std")
    es_strict_mean = summary_payload["aggregate"].get("val/error_sentence_accuracy_strict_mean")
    es_strict_std = summary_payload["aggregate"].get("val/error_sentence_accuracy_strict_std")

    print(
        f"\nAggregated metrics over {len(per_run)} runs:\n"
        f"  Error flag accuracy: {_fmt_pair(ef_mean, ef_std)}\n"
        f"  Error sentence accuracy: {_fmt_pair(es_mean, es_std)}\n"
        f"  Error sentence accuracy (strict): {_fmt_pair(es_strict_mean, es_strict_std)}"
    )
    print(f"Saved summary → {summary_path}")

    if wandb_run is not None and wandb_module is not None:
        try:
            table = wandb_module.Table(
                columns=
                [
                    "run",
                    "seed",
                    "error_flag_accuracy",
                    "error_sentence_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                    "fpr",
                    "balanced_accuracy",
                    "mcc",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "detection_by_error_type",
                ]
            )
            for entry in per_run:
                det_json = json.dumps(entry.get("detection_by_error_type", {}))
                confusion = entry.get("confusion_counts", {}) or {}
                table.add_data(
                    entry["run"],
                    entry["seed"],
                    entry.get("error_flag_accuracy"),
                    entry.get("error_sentence_accuracy"),
                    entry.get("precision"),
                    entry.get("recall"),
                    entry.get("f1"),
                    entry.get("specificity"),
                    entry.get("fpr"),
                    entry.get("balanced_accuracy"),
                    entry.get("mcc"),
                    confusion.get("tp"),
                    confusion.get("fp"),
                    confusion.get("tn"),
                    confusion.get("fn"),
                    det_json,
                )
            wandb_module.log({"val/per_run": table})

            wandb_run.summary["n_val_examples"] = summary_payload["aggregate"]["n_val_examples"]
            wandb_run.summary["summary_json"] = str(summary_path)
            for metric_name, stats_dict in aggregate_metrics.items():
                wandb_run.summary[f"{metric_name}_mean"] = stats_dict.get("mean")
                wandb_run.summary[f"{metric_name}_std"] = stats_dict.get("std")
            if aggregate_by_type:
                wandb_run.summary["val/by_error_type"] = aggregate_by_type
        except Exception:
            pass
        finally:
            try:
                wandb_module.finish()
            except Exception:
                pass

    if args.debug:
        debug_path = run_dir / "debug_summary.json"
        debug_path.write_text(json.dumps(debug_rows, indent=2) + "\n", encoding="utf-8")
        print(f"Saved debug preview → {debug_path}")


if __name__ == "__main__":
    main()
