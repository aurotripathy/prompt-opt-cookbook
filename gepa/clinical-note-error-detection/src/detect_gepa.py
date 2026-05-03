"""GEPA optimisation of the MEDIQA-CORR detection prompt (paper baseline)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dataset import load_csv, stratified_first_k, to_examples
from models import PRESET_CHOICES, build_lm_from_preset
from task_utils import (
    _split_sentence_idx_and_text,
    format_sentences_for_paper_prompt,
    parse_paper_prompt_output,
    set_seed,
    short_id,
)
from evaluation import (
    _clean_value,
    _plot_binary_confusion,
    evaluate_programme,
    gepa_detection_metric,
    save_result_outputs,
)


# ---------------------------------------------------------------------------
# DSPy programme (paper baseline)

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

    class PaperPromptSignature(dspy.Signature):
        __doc__ = PAPER_PROMPT_TEXT  # the seed prompt text, gepa mutates it.

        input: str = dspy.InputField()
        output: str = dspy.OutputField()

    try:
        PaperPromptSignature.model_rebuild()  # make sure the Signature's compiled schema reflects the __doc__ I just assigned"
    except Exception:
        pass

    class PaperPromptProgramme(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            # Given the Signature, (I fields + O fields + Instructions), 
            # build the prompt, and call the model to get the response into the Output field.
            self.detect = dspy.Predict(PaperPromptSignature)

        def forward(self, sentences: List[str]) -> dspy.Prediction:
            # Builds messages from PaperPromptSignature.instructions + the formatted sentences.
            # Sends the request, gets a completion, parses it.
            # Returns a dspy.Prediction with .output matching the signature's OutputField.
            formatted = format_sentences_for_paper_prompt(sentences)  # The formatted sentences are passed to the model.
            completion = self.detect(input=formatted)
            raw_response = str(getattr(completion, "output", "")).strip()

            # post-process the raw response to get the verdict, sentence id, and correction.
            verdict, parsed_sentence_id, parsed_correction = parse_paper_prompt_output(
                raw_response,
            )
            normalized_verdict = "error" if verdict.lower() == "error" else "correct"

            sentence_id_str: Optional[str] = None
            if parsed_sentence_id is not None:
                sentence_id_str = str(parsed_sentence_id)

            return dspy.Prediction(
                verdict=normalized_verdict,
                raw_response=raw_response,
                predicted_sentence_id=sentence_id_str,
                corrected_sentence=parsed_correction,
            )

except ModuleNotFoundError as exc:  # pragma: no cover - validated at runtime
    dspy = None  # type: ignore
    Literal = None  # type: ignore  # noqa: N816
    DSPY_IMPORT_ERROR = exc
else:
    DSPY_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# GEPA driver

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv", 
        default="data/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv")
    parser.add_argument(
        "--val-csv",
        default="data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
    )
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--gepa-train-k", type=int, default=0, help="Optional stratified subset for GEPA train")
    parser.add_argument(
        "--gepa-pareto-k",
        type=int,
        default=0,
        help="Optional stratified subset for GEPA Pareto tracking (0 = full val set).",
    )
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="qwen3-8b")
    parser.add_argument(
        "--reflector-preset",
        choices=PRESET_CHOICES,
        default=None,
        help="Reflection LM preset (defaults to --preset).",
    )
    parser.add_argument(
        "--reflector-port",
        type=int,
        default=None,
        help="Optional port for the reflector LM when using local models (defaults to --port).",
    )
    parser.add_argument("--runs", type=int, default=1, help="Evaluation repeats after GEPA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, dest="top_p", default=None)
    parser.add_argument("--top-k", type=int, dest="top_k", default=None)
    parser.add_argument("--min-p", type=float, dest="min_p", default=None)
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", default=None)
    parser.add_argument("--port", type=int, default=7501)
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA auto budget preset (ignored if --max-metric-calls or --max-full-evals set)",
    )
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--max-full-evals", type=float, default=None)
    parser.add_argument("--output-dir", default="results/gepa")
    parser.add_argument("--save-program", default=None)
    parser.add_argument("--load-program", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="medec-detect-gepa-p1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reflector_port is None:
        args.reflector_port = args.port

    if DSPY_IMPORT_ERROR is not None:
        raise SystemExit(
            "DSPy is required for this script but could not be imported: "
            f"{DSPY_IMPORT_ERROR}",
        )

    set_seed(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df_train = (
        load_csv(args.train_csv)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    df_val = (
        load_csv(args.val_csv)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    if args.limit_train and args.limit_train > 0:
        df_train = df_train.head(args.limit_train)
    if args.limit_val and args.limit_val > 0:
        df_val = df_val.head(args.limit_val)

    trainset_full = to_examples(df_train)  # dspy.Example objects.
    valset_full = to_examples(df_val)  # dspy.Example objects.

    trainset = stratified_first_k(trainset_full, int(args.gepa_train_k)) if args.gepa_train_k else trainset_full
    pareto_valset = stratified_first_k(valset_full, int(args.gepa_pareto_k)) if args.gepa_pareto_k else valset_full

    if not trainset:
        raise SystemExit("GEPA train set is empty; adjust --limit-train or --gepa-train-k.")
    if not pareto_valset:
        raise SystemExit("GEPA pareto set is empty; adjust --limit-val or --gepa-pareto-k.")

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

    reflector_preset = args.reflector_preset or args.preset
    reflector_lm, reflector_run_name, reflector_info = build_lm_from_preset(
        reflector_preset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        port=args.reflector_port,
        configure=False,
    )

    ts = dt.datetime.now().strftime("%Y_%m_%d-%H%M")
    suffix = short_id()
    run_dir = out_root / (
        f"gepa_detect_{run_name}_ref-{reflector_run_name}_seed{args.seed}_{ts}_{suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_module = None
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore

            wandb_module = wandb
            wandb_run_name = args.wandb_run_name or f"{run_name}-gepa-{ts}-{suffix}"
            wandb_config = {
                "preset": args.preset,
                "reflector": reflector_preset,
                "auto": args.auto,
                "seed": int(args.seed),
                "gepa_train": len(trainset),
                "gepa_pareto": len(pareto_valset),
                "lm": lm_info,
                "reflector_lm": reflector_info,
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
    print(f"  Inference preset: {args.preset}")
    print(f"  Reflector preset: {reflector_preset}")
    print(f"  Train examples (full/GEPA): {len(trainset_full)}/{len(trainset)}")
    print(f"  Val examples (full/Pareto): {len(valset_full)}/{len(pareto_valset)}")
    print(
        "  GEPA budgets → auto={auto} max_metric_calls={mmc} max_full_evals={mfe}".format(
            auto=args.auto if args.max_metric_calls is None and args.max_full_evals is None else None,
            mmc=args.max_metric_calls,
            mfe=args.max_full_evals,
        )
    )
    print(f"  Output directory: {run_dir}\n")

    programme = PaperPromptProgramme()

    gepa_budget_metric_calls: Optional[int] = None
    gepa_budget_full_evals: Optional[float] = None
    gepa_actual_metric_calls: Optional[int] = None
    gepa_actual_full_evals: Optional[float] = None
    gepa_actual_full_val_evals: Optional[int] = None
    gepa_log_dir: Optional[str] = None

    baseline_result: Dict[str, Any] = {}
    baseline_paths: Dict[str, Optional[str]] = {}

    def run_baseline_evaluation() -> None:
        nonlocal baseline_result, baseline_paths
        baseline_result = evaluate_programme(programme, valset_full)
        baseline_paths = save_result_outputs(run_dir, "val", "baseline", baseline_result)
        if wandb_run is not None and wandb_module is not None:
            ef_acc = baseline_result.get("error_flag_accuracy")
            if ef_acc is not None and not math.isnan(ef_acc):
                wandb_module.log({"val/baseline_error_flag_accuracy": ef_acc})

    optimised_program = None

    if args.load_program:
        print(f"Loading compiled programme state from {args.load_program}")
        optimised_program = PaperPromptProgramme()
        try:
            optimised_program.load(args.load_program)
            print(f"Loaded compiled programme state ← {args.load_program}")
        except Exception as exc:
            raise SystemExit(f"Failed to load programme from {args.load_program}: {exc}")
    else:
        run_baseline_evaluation()

        try:
            from dspy import GEPA  # type: ignore

            if args.auto and (args.max_metric_calls is not None or args.max_full_evals is not None):
                raise SystemExit("Specify only one of --auto, --max-metric-calls, or --max-full-evals.")

            gepa_log_dir_path = run_dir / "gepa_log"
            gepa_log_dir_path.mkdir(parents=True, exist_ok=True)

            gepa_kwargs: Dict[str, Any] = {
                "metric": gepa_detection_metric,
                "track_stats": True,
                "track_best_outputs": True,
                "add_format_failure_as_feedback": True,
                "reflection_lm": reflector_lm,
                "use_wandb": args.wandb,
                "seed": args.seed,
                "log_dir": str(gepa_log_dir_path),
            }
            if args.max_metric_calls is not None:
                gepa_kwargs["max_metric_calls"] = int(args.max_metric_calls)
            elif args.max_full_evals is not None:
                gepa_kwargs["max_full_evals"] = float(args.max_full_evals)
            else:
                gepa_kwargs["auto"] = args.auto

            gepa = GEPA(**gepa_kwargs)
            optimised_program = gepa.compile(programme, trainset=trainset, valset=pareto_valset)

            try:
                budget_val = getattr(gepa, "max_metric_calls", None)
                if isinstance(budget_val, (int, float)):
                    gepa_budget_metric_calls = int(budget_val)
                    denom = len(trainset) + len(pareto_valset)
                    if denom > 0:
                        gepa_budget_full_evals = float(gepa_budget_metric_calls / denom)
            except Exception:
                pass

            try:
                detailed = getattr(optimised_program, "detailed_results", None)
                if detailed is not None:
                    tm_calls = getattr(detailed, "total_metric_calls", None)
                    if isinstance(tm_calls, (int, float)):
                        gepa_actual_metric_calls = int(tm_calls)
                        denom = len(trainset) + len(pareto_valset)
                        if denom > 0:
                            gepa_actual_full_evals = float(gepa_actual_metric_calls / denom)
                    num_full_val = getattr(detailed, "num_full_val_evals", None)
                    if isinstance(num_full_val, (int, float)):
                        gepa_actual_full_val_evals = int(num_full_val)
                    log_dir = getattr(detailed, "log_dir", None)
                    if isinstance(log_dir, str):
                        gepa_log_dir = log_dir
            except Exception:
                pass

            save_path = Path(args.save_program) if args.save_program else (run_dir / "program.json")
            try:
                if hasattr(optimised_program, "save"):
                    optimised_program.save(str(save_path))
                    print(f"Saved compiled programme → {save_path}")
            except Exception as exc:
                print(f"Warning: failed to save programme ({exc})")

        except Exception as exc:
            raise SystemExit(f"GEPA optimisation failed: {exc}") from exc

    if optimised_program is None:
        raise SystemExit("No optimised programme is available for evaluation.")

    seeds = [args.seed + i for i in range(args.runs)]
    per_run_results: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    optimised_paths: Dict[str, Optional[str]] = {}

    for idx, run_seed in enumerate(seeds, 1):
        print(f"=== Evaluation repeat {idx}/{args.runs} seed={run_seed} ===")
        set_seed(run_seed)
        build_lm_from_preset(
            args.preset,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=run_seed,
            port=args.port,
            configure=True,
        )
        result = evaluate_programme(optimised_program, valset_full)
        ef_acc = float(result.get("error_flag_accuracy", float("nan")))
        if math.isnan(ef_acc):
            print("  Error flag accuracy: nan")
        else:
            print(f"  Error flag accuracy: {ef_acc:.3f}")

        run_dir_i = run_dir / f"rep{idx}"
        run_dir_i.mkdir(parents=True, exist_ok=True)
        optimised_paths = save_result_outputs(run_dir_i, "val", f"optimised_rep{idx}", result)

        per_run_results.append(
            {
                "run": idx,
                "seed": run_seed,
                "error_flag_accuracy": _clean_value(result.get("error_flag_accuracy")),
                "precision": _clean_value(result["binary_metrics"].get("precision")),
                "recall": _clean_value(result["binary_metrics"].get("recall")),
                "f1": _clean_value(result["binary_metrics"].get("f1")),
                "balanced_accuracy": _clean_value(result["binary_metrics"].get("balanced_accuracy")),
                "mcc": _clean_value(result["binary_metrics"].get("mcc")),
                "confusion_counts": result.get("confusion_counts"),
            }
        )

        if args.debug:
            debug_rows.append(
                {
                    "run": idx,
                    "seed": run_seed,
                    "rows": result.get("rows", [])[:5],
                }
            )

        if wandb_run is not None and wandb_module is not None:
            payload = {
                "run/index": idx,
                "run/seed": run_seed,
            }
            if not math.isnan(ef_acc):
                payload["val/error_flag_accuracy"] = ef_acc
            wandb_module.log(payload)

    metric_values = [r["error_flag_accuracy"] for r in per_run_results if r.get("error_flag_accuracy") is not None]
    if metric_values:
        mean_val = float(stats.mean(metric_values))
        std_val = float(stats.pstdev(metric_values)) if len(metric_values) > 1 else 0.0
    else:
        mean_val = float("nan")
        std_val = float("nan")

    summary_payload: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "preset": args.preset,
        "reflector_preset": reflector_preset,
        "seed": int(args.seed),
        "runs": int(args.runs),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "lm_info": lm_info,
        "reflector_info": reflector_info,
        "gepa_train_size": len(trainset),
        "gepa_pareto_size": len(pareto_valset),
        "baseline": {
            "error_flag_accuracy": _clean_value(baseline_result.get("error_flag_accuracy")),
            "binary_metrics": {
                k: _clean_value(v) for k, v in baseline_result.get("binary_metrics", {}).items()
            },
            "outputs": baseline_paths,
        },
        "optimised": {
            "per_run": per_run_results,
            "mean_error_flag_accuracy": _clean_value(mean_val),
            "std_error_flag_accuracy": _clean_value(std_val),
            "outputs_last_run": optimised_paths,
        },
        "gepa_budget": {
            "auto": args.auto,
            "max_metric_calls": _clean_value(gepa_budget_metric_calls),
            "max_full_evals": _clean_value(gepa_budget_full_evals),
            "actual_metric_calls": _clean_value(gepa_actual_metric_calls),
            "actual_full_evals": _clean_value(gepa_actual_full_evals),
            "actual_full_val_evals": _clean_value(gepa_actual_full_val_evals),
            "log_dir": gepa_log_dir,
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary → {summary_path}")

    if args.debug:
        debug_path = run_dir / "debug_preview.json"
        debug_path.write_text(json.dumps(debug_rows, indent=2) + "\n", encoding="utf-8")
        print(f"Saved debug preview → {debug_path}")

    if wandb_run is not None and wandb_module is not None:
        try:
            wandb_run.summary["baseline_error_flag_accuracy"] = baseline_result.get("error_flag_accuracy")
            wandb_run.summary["optimised_error_flag_accuracy_mean"] = _clean_value(mean_val)
            wandb_run.summary["optimised_error_flag_accuracy_std"] = _clean_value(std_val)
            if gepa_actual_metric_calls is not None:
                wandb_run.summary["gepa_metric_calls"] = gepa_actual_metric_calls
            if gepa_actual_full_evals is not None:
                wandb_run.summary["gepa_full_evals"] = gepa_actual_full_evals
            if gepa_actual_full_val_evals is not None:
                wandb_run.summary["gepa_full_val_evals"] = gepa_actual_full_val_evals
        except Exception:
            pass
        finally:
            try:
                wandb_module.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
