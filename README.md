# ft-rl-platform-benchmark — results

Generated charts and metrics for the HolboxAI fine-tuning / RL platform
benchmark. The source repo is
[`HolboxAI/ft-rl-platform-benchmark`](https://github.com/HolboxAI/ft-rl-platform-benchmark).

Every chart is a PNG. Every chart lives next to a `summary.json` that holds the
numbers behind it. The raw source files are mirrored under `raw/`.

## Folder layout

```
summary.json                     master index of every run and chart
README.md
<platform>/<experiment>/
    *.png                        one or more charts
    summary.json                 metrics + the series behind the charts
cross-platform/
    *.png                        overlays across platforms
    summary.json
raw/<source-relative-path>       verbatim copies of the source metric files
generate_all.py                  rebuilds everything from the source repo
```

## Tasks

- **ORena** — surgical VQA (SFT + GRPO on Qwen3.5-9B / Qwen3-VL-8B).
- **GoEmotions** — multi-label emotion (SFT + GRPO on Qwen3-0.6B).
- **CholecT50** — surgical action-triplet (SFT + GRPO on Qwen3.5-4B VLM).
- **DS-1000** — code generation (GRPO side experiment).

## Platforms

Fireworks, SageMaker, Prime Intellect, Miles, Tinker, verl, SkyRL, Trainium2,
Baseten, Together AI, verl-TPU (Cloud TPU), TitanRL.

Baseten and Together AI produced no data (blocked / not feasible). SageMaker RL
and Prime Intellect ORena produced no learning curve (reward stayed 0 or blocked).
These runs still have a `summary.json` that records their state.

## Charts

Per-run charts:

- `loss_vs_step.png` — train cross-entropy loss (SFT runs, and some RL).
- `reward_vs_step.png` / `score_vs_step.png` — RL reward (Jaccard, exact match,
  blended reward, or pass rate).
- `component_f1_vs_step.png` — CholecT50 triplet / instrument / verb / target F1.
- `before_after.png` — Tinker base vs SFT vs RL evaluation.

Cross-platform charts (in `cross-platform/`):

- `goemotions_rl_jaccard.png` — Jaccard by training progress (Fireworks, Miles, PI).
- `orena_rl_exact_match.png` — exact match by progress (Fireworks, Miles).
- `cholect50_reward_curves.png` — blended reward by progress (verl, Miles, SkyRL, Tinker).
- `cholect50_val_reward.png` — SFT baseline vs RL final.
- `cost_usd.png`, `wall_time_s.png` — cost and wall time by run.

## Rebuild

```
python generate_all.py --src ../ft-rl-platform-benchmark
```

Requires Python 3 with `matplotlib` and `numpy`.
