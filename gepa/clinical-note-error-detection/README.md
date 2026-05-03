# Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models

> **Attribution.** This directory is a refactored fork of
> [`CraigMyles/clinical-note-error-detection`](https://github.com/CraigMyles/clinical-note-error-detection)
> (commit [`a5c9820`](https://github.com/CraigMyles/clinical-note-error-detection/commit/a5c9820))
> by Craig Myles, Patrick Schrempf, and David Harris-Birtill, released under the
> [MIT License](./LICENSE) (© 2026 Craig Myles). All experimental methodology, prompts,
> dataset handling, and benchmark numbers are theirs and unchanged. The modifications
> in this fork are housekeeping only: deduplicating helpers into `src/task_utils.py`,
> tightening LM construction in `src/models.py` (cache off by default, retry kwarg,
> `.env` fallback for API keys, dropping unsupported provider params), removing dead
> training-related argparse from `src/detect_eval.py`, wiring `log_dir` and a
> Pareto-evolution plot (`src/plot_pareto_front.py`) into the GEPA flow, and the
> related README updates. No scoring or optimisation behaviour was changed.

[Paper](https://arxiv.org/abs/2602.22483) | [Original repo](https://github.com/CraigMyles/clinical-note-error-detection) | [MEDEC Dataset](https://github.com/abachaa/MEDEC) | [Citation](#citation)

## Summary

This repository provides reproducibility code for the paper *"Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models"*.

Errors in medical text can cause delays or even result in incorrect treatment for patients. We explore the importance of prompt optimisation for small and large language models applied to the task of error detection in clinical notes, performing rigorous experiments across frontier models (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, Grok 4) and open-source models (Qwen3 0.6B-32B). We show that automatic prompt optimisation with Genetic-Pareto (GEPA) improves error detection accuracy from 0.669 to 0.785 with GPT-5 and from 0.578 to 0.690 with Qwen3-32B, approaching the performance of medical doctors and achieving state-of-the-art on the MEDEC benchmark.

---



---

## Overview

The pipeline consists of three phases:


| Phase               | Script                              | Description                                                               |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------- |
| 1. Baseline         | `src/detect_eval.py`                | Single-pass inference with the paper prompt                               |
| 2. GEPA Compilation | `src/detect_gepa.py --auto heavy`   | Compile optimised prompts on the validation set (produces `program.json`) |
| 3. GEPA Evaluation  | `src/detect_gepa.py --load-program` | Evaluate compiled programmes on held-out test sets                        |


---

## Setup

```bash
uv sync   # or: pip install .
```

Set whichever API keys you need:

```bash
export OPENAI_API_KEY="..."        # GPT-5
export OPENROUTER_API_KEY="..."    # Claude, Gemini, Grok, DeepSeek via OpenRouter
export GOOGLE_API_KEY="..."        # Gemini direct (gemini-3.1-pro-preview, gemini-2.5-pro-direct, gemini-1.5-pro)
export WANDB_API_KEY="..."         # Experiment tracking (optional)
```

Local models are served via [SGLang](https://github.com/sgl-project/sglang):

```bash
python -m sglang.launch_server --port 7501 --model-path Qwen/Qwen3-8B --tp 4
```

## Dataset

The [MEDEC dataset](https://github.com/abachaa/MEDEC) is from the MEDIQA-CORR 2024 shared task. We use the **original** dataset (not the [corrected version](https://github.com/abachaa/MEDEC/blame/70268d24e3ce0cd6d0e099ff7bfd4966f2bbcc28/README.md#L49)) to ensure direct comparability with previous benchmarks.

- **MEDEC-MS** — publicly available
- **MEDEC-UW** — requires a Data Use Agreement (see [dataset repo](https://github.com/abachaa/MEDEC) for details)

### Data layout

The scripts read CSVs from a local `data/` directory (gitignored). The simplest way to populate it is to clone the dataset repo into `data/MEDEC/`:

```bash
cd gepa/clinical-note-error-detection
git clone https://github.com/abachaa/MEDEC data/MEDEC
```

Resulting structure:

```
gepa/clinical-note-error-detection/
└── data/
    └── MEDEC/
        ├── README.md
        └── MEDEC-MS/
            ├── MEDEC-Full-TrainingSet-with-ErrorType.csv
            ├── MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv
            └── MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv
```

This matches the argparse defaults of `detect_eval.py` and `detect_gepa.py`:

```bash
--train-csv data/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv
--val-csv   data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv
```

To evaluate against the held-out test set, override `--val-csv` (the flag is reused for the eval split):

```bash
--val-csv   data/MEDEC/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv
```

## Usage

**Baseline inference:**

```bash
# Local model
python src/detect_eval.py \
  --preset qwen3-8b \              # model preset (see Supported Models)
  --port 7501 \                    # SGLang server port
  --prompt paper \                 # prompt style matching the paper
  --runs 3 \                       # independent seeded repeats
  --seed 42 \                      # base random seed
  --eval-csv data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
  --output-dir results/baseline \
  --wandb                          # enable W&B logging (optional)

# API model
python src/detect_eval.py \
  --preset gpt-5 \                 # uses OpenAI API directly
  --prompt paper \
  --runs 3 \
  --seed 42 \
  --eval-csv data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
  --output-dir results/baseline \
  --wandb
```

> `detect_eval.py` accepts `--eval-csv` (canonical) and `--val-csv` (alias kept for backward compatibility) interchangeably; same for `--limit-eval` / `--limit-val`. The flag points at any split — validation, test, or a held-out probe — since the script is single-pass inference and never trains.

**GEPA compilation** (Phase 2):

```bash
python src/detect_gepa.py \
  --preset qwen3-8b \              # inference model
  --reflector-preset qwen3-32b \   # reflector model used by GEPA optimiser
  --port 7501 \                    # SGLang port for inference model
  --reflector-port 7502 \          # SGLang port for reflector model
  --auto heavy \                   # GEPA budget (light/medium/heavy)
  --runs 1 \
  --seed 42 \
  --val-csv data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
  --output-dir results/gepa_grid \
  --wandb
```

**GEPA compilation with Gemini direct API** (no local SGLang server, requires `GOOGLE_API_KEY`):

```bash
python src/detect_gepa.py \
  --preset gemini-2.5-flash-lite \
  --reflector-preset gemini-2.5-pro-direct \
  --auto heavy \
  --val-csv data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
  --output-dir results/gepa_gemini
```

> **Smoke-test first.** Gemini API calls are billed per token; before kicking off `--auto heavy` on the full split, validate the wiring with a tiny subset using `--limit-train` and `--limit-val`:
>
> ```bash
> python src/detect_gepa.py \
>   --preset gemini-2.5-flash-lite \
>   --reflector-preset gemini-2.5-pro-direct \
>   --auto light \
>   --limit-train 100 \
>   --limit-val 60 \
>   --val-csv data/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
>   --output-dir results/gepa_gemini_smoke
> ```
>
> `--limit-train N` and `--limit-val N` cap the number of rows loaded from each CSV (`0` = use all, the default). They work with any preset, but are most useful for paid APIs where a 1000-row `--auto heavy` run can be expensive.

**GEPA test evaluation** (Phase 3):

```bash
python src/detect_gepa.py \
  --preset qwen3-8b \
  --reflector-preset qwen3-32b \
  --port 7501 \
  --reflector-port 7502 \
  --seed 42 \
  --val-csv data/MEDEC/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv \
  --output-dir results/gepa_test \
  --load-program results/gepa_grid/.../program.json \  # compiled programme from Phase 2
  --wandb
```

## Diagnostics: Pareto-front evolution plot

Each `detect_gepa.py` compilation run writes GEPA's internal optimiser state to `<run_dir>/gepa_log/gepa_state.bin` (a pickled snapshot of every candidate prompt, per-example valset scores, parent lineage, and the per-iteration Pareto front). `src/plot_pareto_front.py` reads that pickle and produces a single diagnostic plot, `pareto_evolution_score.png`, that tracks how the Pareto-front aggregate score grows as new prompt candidates are introduced.

```bash
python src/plot_pareto_front.py \
  --state results/gepa_gemini/<run_dir>/gepa_log/gepa_state.bin \
  --out   results/gepa_gemini/<run_dir>/pareto/
```

The plot has two curves on a shared y-axis (aggregate valset score, 0–1):

- **Pareto-front aggregate** (green, monotonic) — best-so-far per valset example, averaged. This is the metric GEPA logs as `Valset pareto front aggregate score` at every iteration.
- **Individual candidate aggregate** (blue X) — that single candidate's own raw valset score. Bounces around because not every proposed prompt is an improvement.

The top axis is annotated with **cumulative metric calls at discovery**, so you can read cost-vs-quality directly: a sharp jump in green for a small bump on the top axis = high marginal value per LLM call; a long flat plateau = candidates that consume budget without moving the front.

## Supported Models


| Preset       | Provider       | Preset                   | Provider           |
| ------------ | -------------- | ------------------------ | ------------------ |
| `qwen3-0.6b` | Local (SGLang) | `gpt-5`                  | OpenAI             |
| `qwen3-1.7b` | Local (SGLang) | `claude-sonnet-4.5`      | OpenRouter         |
| `qwen3-4b`   | Local (SGLang) | `gemini-2.5-pro`         | OpenRouter         |
| `qwen3-8b`   | Local (SGLang) | `grok-4`                 | OpenRouter         |
| `qwen3-14b`  | Local (SGLang) | `deepseek-r1`            | OpenRouter         |
| `qwen3-32b`  | Local (SGLang) | `gemini-3.1-pro-preview` | Google AI (direct) |
|              |                | `gemini-2.5-pro-direct`  | Google AI (direct) |
|              |                | `gemini-1.5-pro`         | Google AI (direct) |
|              |                | `gemini-2.5-flash-lite`  | Google AI (direct) |
|              |                | `gemini-2.5-flash`       | Google AI (direct) |
|              |                | `gemini-2.0-flash`       | Google AI (direct) |


> All `gemini-*` presets except `gemini-2.5-pro` call the Google AI API directly via LiteLLM (`gemini/...`) and require `GOOGLE_API_KEY`. The pre-existing `gemini-2.5-pro` preset still routes through OpenRouter.

## SLURM Scripts

We have additionally provided the SLURM scripts which may be of interest to those trying to reproduce this on HPC environments. The `slurm/` directory contains:

- `**run_medec_qwen3_array.sbatch**` -- Baseline across all 6 Qwen3 sizes (array 0--5)
- `**run_medec_gepa_grid.sbatch**` -- Full 28-job reflector x inference grid with intelligent GPU splitting (1 model: TP=4 on 4 GPUs; 2 models: TP=2 each on 2+2 GPUs)
- `**run_medec_gepa_test_inference.sbatch**` -- Load compiled programmes and evaluate on test sets

All scripts auto-launch SGLang, poll for readiness, and clean up on exit. Override defaults via environment variables:

```bash
sbatch --export=ALL,REPO_ROOT=/path/to/repo slurm/run_medec_gepa_grid.sbatch
```

## Citation

Accepted at [HeaLing](https://healing-workshop.github.io/) @ [EACL 2026](https://2026.eacl.org/).

```bibtex
@article{myles2026importance,
  title={Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models},
  author={Myles, Craig and Schrempf, Patrick and Harris-Birtill, David},
  journal={arXiv preprint},
  doi={10.48550/arXiv.2602.22483},
  url={https://arxiv.org/abs/2602.22483},
  year={2026}
}
```

