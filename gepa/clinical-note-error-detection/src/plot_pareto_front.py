"""Plot the GEPA Pareto-front aggregate score over the candidate timeline.

Reads ``gepa_state.bin`` (produced by GEPA when ``log_dir`` is set) and renders
``pareto_evolution_score.png``: the Pareto-front aggregate score (best-so-far
per valset example, averaged) and each candidate's own aggregate score, plotted
against the candidate index. The top axis annotates cumulative metric calls
spent at each discovery so cost-vs-quality is readable in one glance.

Usage::

    python -m plot_pareto_front --state <run_dir>/gepa_log/gepa_state.bin \\
        --out <run_dir>/pareto/
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_state(state_path: Path) -> dict:
    with state_path.open("rb") as fh:
        state = pickle.load(fh)
    if not isinstance(state, dict) or "prog_candidate_val_subscores" not in state:
        raise SystemExit(
            f"{state_path} doesn't look like a GEPAState pickle "
            "(missing 'prog_candidate_val_subscores')."
        )
    return state


def _reconstruct_evolution(
    state: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the candidate timeline.

    Returns three arrays of length n_candidates:
      * pareto_aggregate[k]  — mean over examples of best score in {0..k}
      * candidate_score[k]   — candidate k's own mean valset score
      * metric_calls[k]      — cumulative metric calls when candidate k was discovered
    """

    subscores = state["prog_candidate_val_subscores"]
    n_candidates = len(subscores)
    example_ids = sorted(subscores[0].keys())
    n_examples = len(example_ids)

    score_matrix = np.zeros((n_candidates, n_examples), dtype=float)
    for k, scores in enumerate(subscores):
        for j, e in enumerate(example_ids):
            score_matrix[k, j] = float(scores.get(e, 0.0))

    candidate_score = score_matrix.mean(axis=1)
    pareto_aggregate = np.maximum.accumulate(score_matrix, axis=0).mean(axis=1)

    metric_calls_raw = state.get("num_metric_calls_by_discovery") or [0] * n_candidates
    metric_calls = np.asarray(metric_calls_raw[:n_candidates], dtype=int)
    return pareto_aggregate, candidate_score, metric_calls


def plot_evolution_score(
    pareto_aggregate: np.ndarray,
    candidate_score: np.ndarray,
    metric_calls: np.ndarray,
    out: Path,
) -> None:
    n = len(pareto_aggregate)
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * n), 4.5))
    ax.plot(
        xs, pareto_aggregate, marker="o", linewidth=2.0, color="seagreen",
        label="Pareto-front aggregate (best-so-far per example, averaged)",
    )
    ax.plot(
        xs, candidate_score, marker="x", linewidth=1.0, color="steelblue",
        alpha=0.8, label="Individual candidate aggregate",
    )
    ax.set_xticks(xs)
    ax.set_xlabel("Candidate index (= order of discovery)")
    ax.set_ylabel("Aggregate valset score")
    ax.set_title("GEPA Pareto-front evolution")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    if metric_calls.any():
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xs)
        ax2.set_xticklabels(
            [str(int(c)) if c else "" for c in metric_calls],
            fontsize=7,
        )
        ax2.set_xlabel("Cumulative metric calls at discovery", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state", type=Path, required=True,
        help="Path to GEPA's pickled gepa_state.bin.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."), help="Output directory.",
    )
    args = parser.parse_args()

    state = _load_state(args.state)
    pareto_aggregate, candidate_score, metric_calls = _reconstruct_evolution(state)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "pareto_evolution_score.png"
    plot_evolution_score(pareto_aggregate, candidate_score, metric_calls, out_path)

    print(f"Candidates: {len(candidate_score)} | examples: {len(state['prog_candidate_val_subscores'][0])}")
    print(f"Final Pareto-front aggregate: {pareto_aggregate[-1]:.3f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
