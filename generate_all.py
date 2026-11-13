#!/usr/bin/env python3
"""Generate every available benchmark graph + a summary JSON per run.

Reads the source repo (default ../ft-rl-platform-benchmark) and writes a
navigable folder tree into this repo's root:

    <platform>/<experiment>/<metric>.png   one or more charts
    <platform>/<experiment>/summary.json   the metrics + series behind the charts
    cross-platform/<metric>.png            overlays across platforms
    cross-platform/summary.json
    raw/<relpath>                          verbatim copies of the source metric files
    summary.json                           master index of every run + chart

Usage:
    python generate_all.py [--src ../ft-rl-platform-benchmark]
"""
import argparse
import ast
import csv
import json
import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Palette (dataviz reference palette — light mode).  Fixed categorical order,
# never cycled.  Sequential = blue.  Ink/grid are the chrome tokens.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TASK_LABEL = {
    "orena": "ORena surgical VQA",
    "goemotions": "GoEmotions",
    "cholec": "CholecT50",
    "ds1000": "DS-1000",
}

# Task inferred from experiment name.
def task_of(name: str) -> str:
    n = name.lower()
    if "orena" in n:
        return "ORena surgical VQA"
    if "goemotion" in n:
        return "GoEmotions"
    if "cholec" in n or "cholect" in n:
        return "CholecT50"
    if "ds1000" in n or "ds_1000" in n:
        return "DS-1000"
    return "unknown"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path) as f:
        return json.load(f)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    return ax


def rolling(xs, window):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def line(ax, x, y, color, label=None, lw=1.8, ms=4, end_label=True, end_off=(4, 0)):
    ax.plot(x, y, lw=lw, color=color, marker="o", ms=ms, label=label,
            markeredgewidth=0, zorder=3)
    if end_label and label is not None and len(y):
        ax.annotate(label, (x[-1], y[-1]), xytext=end_off, textcoords="offset points",
                    color=color, fontsize=9, va="center", weight="bold")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {os.path.relpath(path)}")


def series(x, y):
    return {"x": list(x), "y": list(y)}


def blank_summary(platform, experiment, method, state, base_model):
    return {
        "platform": platform,
        "experiment": experiment,
        "task": task_of(experiment),
        "method": method,
        "state": state,
        "base_model": base_model,
        "series": {},
        "metrics": {},
        "graphs": [],
        "source_files": [],
    }


def norm_x(n):
    """Fraction-of-training x axis for cross-platform overlays."""
    if n <= 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


# --------------------------------------------------------------------------- #
# Parsers for raw logs
# --------------------------------------------------------------------------- #
DICT_RE = re.compile(r"\{'loss': '[^']*'[^}]*\}")


def parse_sagemaker_log(path):
    """Extract (step, loss, mean_token_accuracy, learning_rate) from TRL logs."""
    steps, loss, acc, lr = [], [], [], []
    with open(path, errors="replace") as f:
        text = f.read()
    for i, m in enumerate(DICT_RE.finditer(text), 1):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        steps.append(i)
        loss.append(float(d.get("loss", 0)))
        acc.append(float(d.get("mean_token_accuracy")) if "mean_token_accuracy" in d else None)
        lr.append(float(d.get("learning_rate")) if "learning_rate" in d else None)
    return steps, loss, acc, lr


def parse_trainium_log(path):
    """Extract (step, loss, reward) from TRL-on-Neuron logs (reward on odd steps)."""
    steps, loss, reward = [], [], []
    with open(path, errors="replace") as f:
        text = f.read()
    for i, m in enumerate(DICT_RE.finditer(text), 1):
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        steps.append(i)
        loss.append(float(d.get("loss", 0)))
        reward.append(float(d.get("reward")) if "reward" in d else None)
    return steps, loss, reward


TITAN_STEP_RE = re.compile(r"Step:\s*(\d+)")
TITAN_LOSS_RE = re.compile(r"loss/mean:\s*([-+0-9.eE]+)")
TITAN_REWARD_RE = re.compile(r"reward/(?:_)?mean:\s*([-+0-9.eE]+)")


def parse_titanrl_log(path):
    steps, loss, reward = [], [], []
    with open(path) as f:
        for line in f:
            if "Step:" not in line:
                continue
            m_step = TITAN_STEP_RE.search(line)
            m_loss = TITAN_LOSS_RE.search(line)
            m_rew = TITAN_REWARD_RE.search(line)
            if m_step is None or m_rew is None:
                continue
            steps.append(int(m_step.group(1)))
            reward.append(float(m_rew.group(1)))
            if m_loss:
                loss.append(float(m_loss.group(1)))
    return steps, loss, reward


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Per-experiment single-run charts
# --------------------------------------------------------------------------- #
def plot_loss_from_jsonl(summary, src, jsonl_rel, out_dir, name="loss_vs_step.png",
                         title=None, extra_series=()):
    """Plot train loss from a fireworks/tinker-style JSONL (step + train/loss or loss)."""
    path = os.path.join(src, jsonl_rel)
    rows = read_jsonl(path)
    steps, loss = [], []
    for r in rows:
        if "step" not in r:
            continue
        val = r.get("train/loss", r.get("loss", r.get("train/ce_loss")))
        if val is None:
            continue
        steps.append(r["step"])
        loss.append(float(val))
    summary["source_files"].append(jsonl_rel)
    summary["series"]["loss"] = series(steps, loss)
    for key, vals in extra_series:
        summary["series"][key] = series(steps, vals)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    if len(loss) > 40:
        ax.plot(steps, loss, lw=0.6, alpha=0.3, color=BLUE, zorder=2)
        w = max(1, len(loss) // 25)
        ax.plot(steps, rolling(loss, w), lw=1.8, color=BLUE, zorder=3)
    else:
        ax.plot(steps, loss, lw=1.8, color=BLUE, marker="o", ms=4, markeredgewidth=0, zorder=3)
    ax.set_title(title or f"{summary['platform']} / {summary['experiment']} — train loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    out = os.path.join(out_dir, name)
    save(fig, out)
    summary["graphs"].append(os.path.basename(out))
    if loss:
        summary["metrics"].setdefault("final_train_loss", round(loss[-1], 4))


def plot_curve(summary, out_dir, name, title, xlabel, ylabel, curves):
    """Generic multi-series line chart; curves = list of (x, y, color, label)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    for x, y, color, label in curves:
        if not y:
            continue
        ys = [v for v in y if v is not None]
        if len(ys) != len(y):
            x = [x[i] for i, v in enumerate(y) if v is not None]
            y = ys
        line(ax, x, y, color, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=9, loc="best")
    out = os.path.join(out_dir, name)
    save(fig, out)
    summary["graphs"].append(os.path.basename(out))


# --------------------------------------------------------------------------- #
# Platform builders
# --------------------------------------------------------------------------- #
def build_fireworks(src, root):
    p = "fireworks"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    exps = d["experiments"]
    out = []
    # SFT loss
    for exp in ("orena_sft", "goemotions_sft"):
        e = exps[exp]
        s = blank_summary(p, exp, e.get("method"), e.get("state"), e.get("hf_model_id") or e.get("base_model"))
        s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
        jsonl = os.path.join("fireworks", "metrics", f"{exp}_metrics.jsonl")
        plot_loss_from_jsonl(s, src, jsonl, os.path.join(root, p, exp),
                             title=f"Fireworks / {exp} — train loss")
        write_json(os.path.join(root, p, exp, "summary.json"), s)
        out.append(s)
    # ORena RL
    e = exps["orena_rl"]
    s = blank_summary(p, "orena_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    s["series"]["score"] = series(range(len(e["score_curve"])), e["score_curve"])
    s["series"]["exact_match"] = series(range(len(e["exact_match_curve"])), e["exact_match_curve"])
    s["source_files"].append("fireworks/metrics/final_results.json")
    plot_curve(s, os.path.join(root, p, "orena_rl"), "score_vs_step.png",
               "Fireworks / orena_rl — score & exact match", "training chunk", "score",
               [(range(len(e["score_curve"])), e["score_curve"], CAT[0], "score"),
                (range(len(e["exact_match_curve"])), e["exact_match_curve"], CAT[1], "exact match")])
    write_json(os.path.join(root, p, "orena_rl", "summary.json"), s)
    out.append(s)
    # GoEmotions RL
    e = exps["goemotions_rl"]
    s = blank_summary(p, "goemotions_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    s["series"]["jaccard"] = series(range(len(e["jaccard_curve"])), e["jaccard_curve"])
    s["series"]["micro_f1"] = series(range(len(e["micro_f1_curve"])), e["micro_f1_curve"])
    s["source_files"].append("fireworks/metrics/final_results.json")
    plot_curve(s, os.path.join(root, p, "goemotions_rl"), "reward_vs_step.png",
               "Fireworks / goemotions_rl — Jaccard & micro-F1", "training chunk", "score",
               [(range(len(e["jaccard_curve"])), e["jaccard_curve"], CAT[0], "Jaccard"),
                (range(len(e["micro_f1_curve"])), e["micro_f1_curve"], CAT[1], "micro-F1")])
    write_json(os.path.join(root, p, "goemotions_rl", "summary.json"), s)
    out.append(s)
    return out


def build_sagemaker(src, root):
    p = "sagemaker"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    exps = d["experiments"]
    out = []
    for exp, logf in (("orena_sft", "orena_sft.log"), ("goemotions_sft", "goemotions_sft.log")):
        e = exps[exp]
        s = blank_summary(p, exp, e.get("method"), e.get("state"), e.get("hf_model_id"))
        s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
        steps, loss, acc, lr = parse_sagemaker_log(os.path.join(src, p, "logs", logf))
        s["series"]["loss"] = series(steps, loss)
        if any(a is not None for a in acc):
            s["series"]["mean_token_accuracy"] = series(steps, acc)
        s["source_files"].append(f"{p}/logs/{logf}")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        style_ax(ax)
        ax.plot(steps, loss, lw=1.8, color=BLUE, marker="o", ms=5, markeredgewidth=0, zorder=3)
        ax.set_title(f"SageMaker / {exp} — train loss")
        ax.set_xlabel("logged step")
        ax.set_ylabel("cross-entropy loss")
        save(fig, os.path.join(root, p, exp, "loss_vs_step.png"))
        s["graphs"].append("loss_vs_step.png")
        if loss:
            s["metrics"]["final_logged_loss"] = round(loss[-1], 4)
        write_json(os.path.join(root, p, exp, "summary.json"), s)
        out.append(s)
    # RL runs: reward stayed 0 — document, no graph.
    for exp in ("orena_rl", "goemotions_rl"):
        e = exps[exp]
        s = blank_summary(p, exp, e.get("method"), e.get("state"), e.get("hf_model_id"))
        s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
        s["metrics"]["note"] = "reward stayed 0 (Qwen3 thinking mode exhausted max_completion_length); no learning curve."
        s["source_files"].append(f"{p}/metrics/final_results.json")
        write_json(os.path.join(root, p, exp, "summary.json"), s)
        out.append(s)
    return out


def build_prime_intellect(src, root):
    p = "prime-intellect"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    exps = d["experiments"]
    out = []
    # SFT loss from CSV
    e = exps["goemotions_sft"]
    s = blank_summary(p, "goemotions_sft", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    steps, loss = [], []
    with open(os.path.join(src, p, "logs", "exp3_loss.csv")) as f:
        for row in csv.reader(f):
            if row:
                steps.append(int(row[0]))
                loss.append(float(row[1]))
    s["series"]["loss"] = series(steps, loss)
    s["source_files"].append(f"{p}/logs/exp3_loss.csv")
    plot_curve(s, os.path.join(root, p, "goemotions_sft"), "loss_vs_step.png",
               "Prime Intellect / goemotions_sft — train loss", "step", "cross-entropy loss",
               [(steps, loss, BLUE, "train loss")])
    s["metrics"]["final_train_loss"] = round(loss[-1], 4)
    write_json(os.path.join(root, p, "goemotions_sft", "summary.json"), s)
    out.append(s)
    # GoEmotions RL jaccard curve
    e = exps["goemotions_rl"]
    s = blank_summary(p, "goemotions_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    s["series"]["jaccard"] = series(range(len(e["jaccard_curve"])), e["jaccard_curve"])
    s["source_files"].append(f"{p}/metrics/final_results.json")
    plot_curve(s, os.path.join(root, p, "goemotions_rl"), "reward_vs_step.png",
               "Prime Intellect / goemotions_rl — Jaccard", "step", "Jaccard",
               [(range(len(e["jaccard_curve"])), e["jaccard_curve"], BLUE, "Jaccard")])
    write_json(os.path.join(root, p, "goemotions_rl", "summary.json"), s)
    out.append(s)
    # blocked / not started — document only
    for exp in ("orena_sft", "orena_rl"):
        e = exps[exp]
        s = blank_summary(p, exp, e.get("method"), e.get("state"), e.get("hf_model_id"))
        s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
        s["source_files"].append(f"{p}/metrics/final_results.json")
        write_json(os.path.join(root, p, exp, "summary.json"), s)
        out.append(s)
    return out


def build_miles(src, root):
    p = "miles"
    out = []
    # GoEmotions RL
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    s = blank_summary(p, "goemotions_rl", d.get("method"), "COMPLETED", d.get("base_model"))
    s["metrics"] = {k: v for k, v in d.items() if not k.endswith("_curve") and k not in ("micro_f1_curve_finish_sample", "jaccard_curve")}
    jc = d["jaccard_curve"]
    s["series"]["jaccard"] = series(range(len(jc)), jc)
    s["series"]["micro_f1_per_sample"] = series(range(len(d["micro_f1_curve_finish_sample"])), d["micro_f1_curve_finish_sample"])
    s["source_files"].append(f"{p}/metrics/final_results.json")
    x = range(len(jc))
    plot_curve(s, os.path.join(root, p, "goemotions_rl"), "reward_vs_step.png",
               "Miles / goemotions_rl — Jaccard", "rollout step", "Jaccard",
               [(x, jc, BLUE, "Jaccard (raw)"),
                (x, rolling(jc, 8), CAT[1], "Jaccard (rolling-8)")])
    write_json(os.path.join(root, p, "goemotions_rl", "summary.json"), s)
    out.append(s)
    # ORena RL
    d = load_json(os.path.join(src, p, "metrics", "orena_rl_final_results.json"))
    s = blank_summary(p, "orena_rl", d.get("method"), d.get("state"), d.get("base_model"))
    s["metrics"] = {k: v for k, v in d.items() if not k.endswith("_curve")}
    for key in ("score_curve", "exact_match_curve", "no_excluded_class_curve"):
        if key in d and d[key]:
            s["series"][key.replace("_curve", "")] = series(range(len(d[key])), d[key])
    s["source_files"].append(f"{p}/metrics/orena_rl_final_results.json")
    plot_curve(s, os.path.join(root, p, "orena_rl"), "score_vs_step.png",
               "Miles / orena_rl — score & exact match", "rollout step", "score",
               [(range(len(d["score_curve"])), d["score_curve"], CAT[0], "score"),
                (range(len(d["exact_match_curve"])), d["exact_match_curve"], CAT[1], "exact match")])
    write_json(os.path.join(root, p, "orena_rl", "summary.json"), s)
    out.append(s)
    # CholecT50 RL
    d = load_json(os.path.join(src, p, "metrics", "cholect50_rl_final_results.json"))
    s = blank_summary(p, "cholect50_rl", d.get("method"), d.get("state"), d.get("base_model"))
    s["metrics"] = {k: v for k, v in d.items() if not k.endswith("_curve")}
    curve_keys = ["rollout_reward_curve", "triplet_f1_curve", "instrument_f1_curve",
                  "verb_f1_curve", "target_f1_curve", "val_reward_curve"]
    for key in curve_keys:
        if key in d and d[key]:
            s["series"][key.replace("_curve", "")] = series(range(len(d[key])), d[key])
    s["source_files"].append(f"{p}/metrics/cholect50_rl_final_results.json")
    plot_curve(s, os.path.join(root, p, "cholect50_rl"), "reward_vs_step.png",
               "Miles / cholect50_rl — rollout reward", "rollout step", "blended reward",
               [(range(len(d["rollout_reward_curve"])), d["rollout_reward_curve"], BLUE, "rollout reward")])
    comp = [("triplet_f1_curve", "triplet F1", CAT[0]),
            ("instrument_f1_curve", "instrument F1", CAT[1]),
            ("verb_f1_curve", "verb F1", CAT[2]),
            ("target_f1_curve", "target F1", CAT[3])]
    plot_curve(s, os.path.join(root, p, "cholect50_rl"), "component_f1_vs_step.png",
               "Miles / cholect50_rl — component F1", "rollout step", "F1",
               [(range(len(d[k])), d[k], c, lbl) for k, lbl, c in comp])
    write_json(os.path.join(root, p, "cholect50_rl", "summary.json"), s)
    out.append(s)
    return out


def build_tinker(src, root):
    p = "tinker"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    exps = d["experiments"]
    out = []
    # SFT loss
    e = exps["cholec_sft"]
    s = blank_summary(p, "cholec_sft", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    rows = read_jsonl(os.path.join(src, p, "metrics", "sft_metrics.jsonl"))
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    s["series"]["loss"] = series(steps, loss)
    s["source_files"].append(f"{p}/metrics/sft_metrics.jsonl")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    ax.plot(steps, loss, lw=0.6, alpha=0.3, color=BLUE, zorder=2)
    w = max(1, len(loss) // 25)
    ax.plot(steps, rolling(loss, w), lw=1.8, color=BLUE, zorder=3)
    ax.set_title("Tinker / cholec_sft — train loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    save(fig, os.path.join(root, p, "cholec_sft", "loss_vs_step.png"))
    s["graphs"].append("loss_vs_step.png")
    s["metrics"]["final_train_loss"] = round(loss[-1], 4)
    write_json(os.path.join(root, p, "cholec_sft", "summary.json"), s)
    out.append(s)
    # RL reward curve
    e = exps["cholec_rl"]
    s = blank_summary(p, "cholec_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    rc = e["reward_curve"]
    s["series"]["reward"] = series(range(len(rc)), rc)
    # full per-step rl metrics
    rows = read_jsonl(os.path.join(src, p, "metrics", "rl_metrics.jsonl"))
    s["series"]["entropy"] = series([r["step"] for r in rows], [r.get("optim/entropy") for r in rows])
    s["series"]["kl_sample_train"] = series([r["step"] for r in rows], [r.get("optim/kl_sample_train_v1") for r in rows])
    s["source_files"] += [f"{p}/metrics/final_results.json", f"{p}/metrics/rl_metrics.jsonl"]
    plot_curve(s, os.path.join(root, p, "cholec_rl"), "reward_vs_step.png",
               "Tinker / cholec_rl — blended reward", "GRPO batch", "blended reward",
               [(range(len(rc)), rc, BLUE, "reward")])
    write_json(os.path.join(root, p, "cholec_rl", "summary.json"), s)
    out.append(s)
    # before/after eval (base -> sft -> rl)
    evals = {k: load_json(os.path.join(src, p, "metrics", f"eval_{k}.json"))
             for k in ("base", "sft", "rl")}
    s2 = blank_summary(p, "cholec_eval", "eval", "COMPLETED", e.get("hf_model_id"))
    s2["series"] = {k: {"labels": ["triplet", "instrument", "verb", "target", "blended"],
                        "values": [v["triplet_f1"], v["instrument_f1"], v["verb_f1"], v["target_f1"], v["blended_reward"]]}
                    for k, v in evals.items()}
    s2["source_files"] += [f"{p}/metrics/eval_{k}.json" for k in ("base", "sft", "rl")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    labels = ["triplet", "instrument", "verb", "target", "blended"]
    xpos = np.arange(len(labels))
    w = 0.26
    for i, (k, color) in enumerate([("base", CAT[2]), ("sft", CAT[0]), ("rl", CAT[1])]):
        vals = [evals[k][l + "_f1"] if l != "blended" else evals[k]["blended_reward"] for l in labels]
        ax.bar(xpos + (i - 1) * w, vals, w, color=color, label=k, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Tinker / CholecT50 — base vs SFT vs RL (val, n=300)")
    ax.set_ylabel("F1 / blended reward")
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(root, p, "cholec_eval", "before_after.png"))
    s2["graphs"].append("before_after.png")
    write_json(os.path.join(root, p, "cholec_eval", "summary.json"), s2)
    out.append(s2)
    return out


def build_verl(src, root):
    p = "verl"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    e = d["experiments"]["cholec_rl"]
    s = blank_summary(p, "cholec_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    rc = e["reward_curve"]
    s["series"]["reward"] = series(range(len(rc)), rc)
    s["source_files"].append(f"{p}/metrics/final_results.json")
    plot_curve(s, os.path.join(root, p, "cholec_rl"), "reward_vs_step.png",
               "verl / cholec_rl — blended reward", "step", "blended reward",
               [(range(len(rc)), rc, BLUE, "reward")])
    write_json(os.path.join(root, p, "cholec_rl", "summary.json"), s)
    return [s]


def build_skyrl(src, root):
    p = "skyrl"
    d = load_json(os.path.join(src, p, "metrics", "final_results.json"))
    e = d["experiments"]["cholec_rl"]
    s = blank_summary(p, "cholec_rl", e.get("method"), e.get("state"), e.get("hf_model_id"))
    s["metrics"] = {k: v for k, v in e.items() if not k.endswith("_curve")}
    rc = e.get("reward_curve") or []
    if rc:
        s["series"]["reward"] = series(range(len(rc)), rc)
    s["source_files"].append(f"{p}/metrics/final_results.json")
    plot_curve(s, os.path.join(root, p, "cholec_rl"), "reward_vs_step.png",
               "SkyRL / cholec_rl — blended reward (collapsed to 0)", "step", "blended reward",
               [(range(len(rc)), rc, BLUE, "reward")])
    write_json(os.path.join(root, p, "cholec_rl", "summary.json"), s)
    return [s]


def build_trainium2(src, root):
    p = "trainium2"
    out = []
    # full epoch
    d = load_json(os.path.join(src, p, "metrics", "final_results_full_epoch.json"))
    e = d["experiments"]["cholect50_grpo_full_1epoch"]
    s = blank_summary(p, "cholect50_grpo_full_1epoch", e.get("method"), e.get("state"), e.get("base_model"))
    s["metrics"] = {k: v for k, v in e.items() if k != "findings" and not isinstance(v, (list, dict))}
    s["metrics"]["findings"] = e.get("findings", [])
    logf = os.path.join(src, p, "logs", "run_grpo_1792850149.log")
    steps, loss, reward = parse_trainium_log(logf)
    s["series"]["loss"] = series(steps, loss)
    rw_steps = [st for st, r in zip(steps, reward) if r is not None]
    rw_vals = [r for r in reward if r is not None]
    s["series"]["reward"] = series(rw_steps, rw_vals)
    s["source_files"].append(f"{p}/logs/run_grpo_1792850149.log")
    plot_curve(s, os.path.join(root, p, "cholect50_grpo_full_1epoch"), "reward_vs_step.png",
               "Trainium2 / cholect50_grpo_full_1epoch — reward", "step", "blended reward",
               [(rw_steps, rw_vals, BLUE, "train reward")])
    plot_curve(s, os.path.join(root, p, "cholect50_grpo_full_1epoch"), "loss_vs_step.png",
               "Trainium2 / cholect50_grpo_full_1epoch — GRPO loss", "step", "loss",
               [(steps, loss, CAT[1], "loss")])
    write_json(os.path.join(root, p, "cholect50_grpo_full_1epoch", "summary.json"), s)
    out.append(s)
    # smoke (4-step) — document, small graph
    d = load_json(os.path.join(src, p, "metrics", "final_results_smoke.json"))
    e = d["experiments"]["cholect50_grpo_smoke"]
    s = blank_summary(p, "cholect50_grpo_smoke", e.get("method"), e.get("state"), e.get("base_model"))
    s["metrics"] = {k: v for k, v in e.items() if k != "findings" and not isinstance(v, (list, dict))}
    s["metrics"]["findings"] = e.get("findings", [])
    logf = os.path.join(src, p, "logs", "run_grpo.log")
    steps, loss, reward = parse_trainium_log(logf)
    s["series"]["loss"] = series(steps, loss)
    rw_steps = [st for st, r in zip(steps, reward) if r is not None]
    rw_vals = [r for r in reward if r is not None]
    s["series"]["reward"] = series(rw_steps, rw_vals)
    s["source_files"].append(f"{p}/logs/run_grpo.log")
    plot_curve(s, os.path.join(root, p, "cholect50_grpo_smoke"), "reward_vs_step.png",
               "Trainium2 / cholect50_grpo_smoke — reward (flat)", "step", "blended reward",
               [(rw_steps, rw_vals, BLUE, "train reward")])
    write_json(os.path.join(root, p, "cholect50_grpo_smoke", "summary.json"), s)
    out.append(s)
    return out


def build_verl_tpu(src, root):
    p = "verl-tpu"
    # Values transcribed from ds1000/TPU_RL_RESULTS.md (per-step critic/score/mean).
    scores = [0.375, 0.375, 0.313, 0.250, 0.297, 0.344, 0.125, 0.094, 0.344, 0.234]
    s = blank_summary(p, "ds1000_rl", "grpo", "succeeded", "Qwen3.5-4B (text-only)")
    s["metrics"] = {"steps": 10, "final_score": 0.234, "wall_time": "~7 min",
                    "dataset": "DS-1000 (800 train / 128 val)"}
    s["series"]["score"] = series(range(1, len(scores) + 1), scores)
    s["source_files"].append("ds1000/TPU_RL_RESULTS.md")
    plot_curve(s, os.path.join(root, p, "ds1000_rl"), "reward_vs_step.png",
               "verl-TPU / ds1000_rl — pass rate", "step", "DS-1000 pass rate",
               [(range(1, len(scores) + 1), scores, BLUE, "score/mean")])
    write_json(os.path.join(root, p, "ds1000_rl", "summary.json"), s)
    return [s]


def build_titanrl(src, root):
    p = "titanrl"
    steps, loss, reward = parse_titanrl_log(os.path.join(src, p, "ds1000", "ds1000_train.log"))
    s = blank_summary(p, "ds1000_grpo", "grpo", "stopped_at_step_25", "Qwen3-0.6B")
    s["metrics"] = {"steps": len(steps), "final_reward": round(reward[-1], 4) if reward else None,
                    "dataset": "DS-1000 (800 train / 200 val, prompts >5000 chars filtered)"}
    s["series"]["reward"] = series(steps, reward)
    if len(loss) == len(steps):
        s["series"]["loss"] = series(steps, loss)
    s["source_files"].append(f"{p}/ds1000/ds1000_train.log")
    plot_curve(s, os.path.join(root, p, "ds1000"), "reward_vs_step.png",
               "TitanRL / ds1000 — reward", "step", "DS-1000 reward",
               [(steps, reward, BLUE, "reward/mean")])
    if len(loss) == len(steps):
        plot_curve(s, os.path.join(root, p, "ds1000"), "loss_vs_step.png",
                   "TitanRL / ds1000 — loss", "step", "GRPO loss",
                   [(steps, loss, CAT[1], "loss/mean")])
    write_json(os.path.join(root, p, "ds1000", "summary.json"), s)
    return [s]


# --------------------------------------------------------------------------- #
# SkyRL on Cloud TPU (skyrl-tpu) — CholecT50 + DS-1000
# Per-step numbers are transcribed from skyrl-tpu/*/README.md; the raw
# metrics.jsonl lives in GCS and is not committed to the source repo.
# --------------------------------------------------------------------------- #
def build_skyrl_tpu_cholect50(src, root):
    p = "skyrl-tpu"
    # GPU run (SkyRL on g5.12xlarge) vs TPU run (SkyRL on v6e-8), steps 1..16.
    gpu = [0.093, 0.092, 0.154, 0.061, 0.011, 0.076, 0.221, 0.181, 0.186, 0.208,
           0.248, 0.181, 0.306, 0.0, 0.0]           # steps 1..15 (step 16 not measured)
    tpu = [0.292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    s = blank_summary(p, "cholect50", "grpo", "collapsed", "Qwen/Qwen3.5-4B")
    s["metrics"] = {
        "hardware": "Cloud TPU v6e-8 (Trillium, 8 chips)",
        "stack": "SkyRL 2c54cefc + skyrl-tpu patch, TorchTPU 0.2.0",
        "val_reward_before": {"gpu": 0.329, "tpu": 0.379},
        "val_reward_after": {"gpu": None, "tpu": 0.0},
        "note": "TPU run collapses after its first update; all 128 rollouts score 0 from step 2.",
    }
    s["series"]["reward_gpu"] = series(range(1, len(gpu) + 1), gpu)
    s["series"]["reward_tpu"] = series(range(1, len(tpu) + 1), tpu)
    s["source_files"] += ["skyrl-tpu/README.md", "skyrl-tpu/config/gpu_reference_final_results.json"]
    plot_curve(s, os.path.join(root, p, "cholect50"), "reward_vs_step_gpu_tpu.png",
               "SkyRL CholecT50 — train reward: GPU vs Cloud TPU", "step", "mean reward (128 rollouts)",
               [(range(1, len(gpu) + 1), gpu, CAT[0], "GPU (g5.12xlarge)"),
                (range(1, len(tpu) + 1), tpu, CAT[1], "TPU (v6e-8)")])
    write_json(os.path.join(root, p, "cholect50", "summary.json"), s)
    return [s]


def build_skyrl_tpu_ds1000(src, root):
    p = "skyrl-tpu"
    # 20-step GRPO run; table from skyrl-tpu/skyrl_ds1000/README.md.
    steps = list(range(1, 21))
    train_reward = [0.156, 0.180, 0.305, 0.203, 0.336, 0.227, 0.211, 0.289, 0.234, 0.383,
                    0.305, 0.328, 0.234, 0.195, 0.320, 0.352, 0.391, 0.289, 0.344, 0.219]
    pass_at_8 = [0.44, 0.44, 0.63, 0.44, 0.69, 0.50, 0.50, 0.63, 0.50, 0.69,
                 0.56, 0.75, 0.63, 0.44, 0.50, 0.63, 0.56, 0.69, 0.56, 0.56]
    kl = [0.000, 0.001, 0.002, 0.003, 0.007, 0.004, 0.007, 0.007, 0.008, 0.010,
          0.020, 0.021, 0.021, 0.021, 0.026, 0.026, 0.039, 0.047, 0.042, 0.054]
    reply_tokens = [155, 82, 116, 161, 85, 128, 131, 58, 103, 104,
                    134, 113, 93, 89, 163, 108, 119, 131, 104, 145]
    minutes = [31.4, 37.7, 35.6, 40.5, 35.6, 39.6, 39.9, 33.3, 39.2, 40.5,
               33.6, 42.4, 37.6, 36.3, 44.8, 40.7, 38.8, 48.1, 39.2, 41.2]
    val_steps = [0, 10, 20]
    val_pass = [0.282, 0.340, 0.350]  # 58/206, 70/206, 72/206

    s = blank_summary(p, "ds1000", "grpo", "completed", "Qwen/Qwen3.5-4B (thinking off)")
    s["metrics"] = {
        "hardware": "Cloud TPU v6e-8 (Trillium, 8 chips)",
        "stack": "SkyRL 2c54cefc + skyrl-tpu patch, TorchTPU 0.2.0, transformers 5.16.1",
        "dataset": "DS-1000 (794 train / 206 val)",
        "grpo": "16 problems x 8 samples/step, lr 1e-6, KL 0.001, replies <= 512 tokens, temp 1.0",
        "val_pass_at_1": {"before": 0.282, "step_10": 0.340, "step_20": 0.350},
        "note": "train reward mean steps 1-5 = 0.236, steps 16-20 = 0.319; no collapse (KL rises slowly).",
    }
    s["series"]["train_reward"] = series(steps, train_reward)
    s["series"]["pass_at_8"] = series(steps, pass_at_8)
    s["series"]["kl"] = series(steps, kl)
    s["series"]["reply_tokens"] = series(steps, reply_tokens)
    s["series"]["minutes_per_step"] = series(steps, minutes)
    s["series"]["val_pass_at_1"] = series(val_steps, val_pass)
    s["source_files"].append("skyrl-tpu/skyrl_ds1000/README.md")

    d = os.path.join(root, p, "ds1000")
    plot_curve(s, d, "reward_vs_step.png",
               "SkyRL DS-1000 on TPU — train reward", "step", "mean reward",
               [(steps, train_reward, BLUE, "train reward (raw)"),
                (steps, rolling(train_reward, 5), CAT[1], "rolling-5")])
    plot_curve(s, d, "pass_at_8_vs_step.png",
               "SkyRL DS-1000 on TPU — pass@8", "step", "share of 16 problems with a pass",
               [(steps, pass_at_8, CAT[0], "pass@8")])
    plot_curve(s, d, "eval_pass_rate.png",
               "SkyRL DS-1000 on TPU — val pass@1", "step", "val pass@1 (206 problems, greedy)",
               [(val_steps, val_pass, CAT[2], "pass@1")])
    # training health small multiples (KL + reply length)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax in axes:
        style_ax(ax)
    line(axes[0], steps, kl, CAT[0], label="KL to reference")
    axes[0].set_ylabel("KL")
    axes[0].set_title("SkyRL DS-1000 on TPU — training health")
    line(axes[1], steps, reply_tokens, CAT[1], label="mean reply tokens")
    axes[1].set_ylabel("reply tokens")
    axes[1].set_xlabel("step")
    out = os.path.join(d, "training_health.png")
    save(fig, out)
    s["graphs"].append(os.path.basename(out))
    write_json(os.path.join(d, "summary.json"), s)
    return [s]


# --------------------------------------------------------------------------- #
# Cross-platform overlays
# --------------------------------------------------------------------------- #
def build_cross_platform(src, root):
    cdir = os.path.join(root, "cross-platform")
    meta = {"charts": {}}

    # --- GoEmotions RL Jaccard (normalized x) ---
    fw = load_json(os.path.join(src, "fireworks", "metrics", "final_results.json"))["experiments"]["goemotions_rl"]["jaccard_curve"]
    mi = load_json(os.path.join(src, "miles", "metrics", "final_results.json"))["jaccard_curve"]
    pi = load_json(os.path.join(src, "prime-intellect", "metrics", "final_results.json"))["experiments"]["goemotions_rl"]["jaccard_curve"]
    curves = [(norm_x(len(fw)), fw, CAT[0], "Fireworks"),
              (norm_x(len(mi)), mi, CAT[1], "Miles"),
              (norm_x(len(pi)), pi, CAT[2], "Prime Intellect")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    for x, y, c, lbl in curves:
        ax.plot(x, y, lw=1.4, color=c, alpha=0.35, zorder=2)
        ax.plot(x, rolling(y, max(1, len(y) // 12)), lw=1.8, color=c, label=lbl, zorder=3)
    ax.set_title("GoEmotions RL — Jaccard by training progress")
    ax.set_xlabel("fraction of training")
    ax.set_ylabel("Jaccard")
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(cdir, "goemotions_rl_jaccard.png"))
    meta["charts"]["goemotions_rl_jaccard"] = {
        "title": "GoEmotions RL — Jaccard by training progress",
        "series": {"Fireworks": series(norm_x(len(fw)), fw), "Miles": series(norm_x(len(mi)), mi),
                   "Prime Intellect": series(norm_x(len(pi)), pi)},
    }

    # --- ORena RL exact match (normalized x) ---
    fw = load_json(os.path.join(src, "fireworks", "metrics", "final_results.json"))["experiments"]["orena_rl"]["exact_match_curve"]
    mi = load_json(os.path.join(src, "miles", "metrics", "orena_rl_final_results.json"))["exact_match_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    for x, y, c, lbl in [(norm_x(len(fw)), fw, CAT[0], "Fireworks"),
                         (norm_x(len(mi)), mi, CAT[1], "Miles")]:
        ax.plot(x, y, lw=1.4, color=c, alpha=0.35, zorder=2)
        ax.plot(x, rolling(y, max(1, len(y) // 10)), lw=1.8, color=c, label=lbl, zorder=3)
    ax.set_title("ORena RL — exact match by training progress")
    ax.set_xlabel("fraction of training")
    ax.set_ylabel("exact match")
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(cdir, "orena_rl_exact_match.png"))
    meta["charts"]["orena_rl_exact_match"] = {
        "title": "ORena RL — exact match by training progress",
        "series": {"Fireworks": series(norm_x(len(fw)), fw), "Miles": series(norm_x(len(mi)), mi)},
    }

    # --- CholecT50 reward curves ---
    verl = load_json(os.path.join(src, "verl", "metrics", "final_results.json"))["experiments"]["cholec_rl"]["reward_curve"]
    miles = load_json(os.path.join(src, "miles", "metrics", "cholect50_rl_final_results.json"))["rollout_reward_curve"]
    skyrl = load_json(os.path.join(src, "skyrl", "metrics", "final_results.json"))["experiments"]["cholec_rl"]["reward_curve"]
    tinker = load_json(os.path.join(src, "tinker", "metrics", "final_results.json"))["experiments"]["cholec_rl"]["reward_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    for x, y, c, lbl in [(norm_x(len(verl)), verl, CAT[0], "verl"),
                         (norm_x(len(miles)), miles, CAT[1], "Miles"),
                         (norm_x(len(skyrl)), skyrl, CAT[2], "SkyRL"),
                         (norm_x(len(tinker)), tinker, CAT[3], "Tinker")]:
        ax.plot(x, y, lw=1.4, color=c, alpha=0.4, zorder=2)
        ax.plot(x, rolling(y, max(1, len(y) // 8)), lw=1.8, color=c, label=lbl, zorder=3)
    ax.set_title("CholecT50 RL — blended reward by training progress")
    ax.set_xlabel("fraction of training")
    ax.set_ylabel("blended reward")
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(cdir, "cholect50_reward_curves.png"))
    meta["charts"]["cholect50_reward_curves"] = {
        "title": "CholecT50 RL — blended reward by training progress",
        "series": {"verl": series(norm_x(len(verl)), verl), "Miles": series(norm_x(len(miles)), miles),
                   "SkyRL": series(norm_x(len(skyrl)), skyrl), "Tinker": series(norm_x(len(tinker)), tinker)},
    }

    # --- CholecT50 val reward before/after ---
    rows = [
        ("Tinker", 0.3642, 0.4188, CAT[4]),
        ("Miles", 0.31875, 0.65125, CAT[1]),
        ("verl", 0.0, 0.715, CAT[0]),
        ("Trainium2", 0.31875, 0.3573, CAT[7]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    xpos = np.arange(len(rows))
    w = 0.36
    ax.bar(xpos - w / 2, [r[1] for r in rows], w, color=[r[3] for r in rows], alpha=0.45,
           label="SFT baseline", zorder=3)
    ax.bar(xpos + w / 2, [r[2] for r in rows], w, color=[r[3] for r in rows], label="RL final", zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("blended reward")
    ax.set_title("CholecT50 — SFT baseline vs RL final val reward")
    ax.legend(frameon=False, fontsize=9)
    save(fig, os.path.join(cdir, "cholect50_val_reward.png"))
    meta["charts"]["cholect50_val_reward"] = {
        "title": "CholecT50 — SFT baseline vs RL final val reward",
        "note": "verl baseline is thinking-disabled SFT (0.0); Tinker/Miles/Trainium2 use thinking-enabled SFT (~0.32).",
        "rows": [{"platform": r[0], "sft_baseline": r[1], "rl_final": r[2]} for r in rows],
    }

    # --- Cost & wall time ---
    cost_rows = collect_scalar(src, "estimated_cost_usd")
    wall_rows = collect_scalar(src, "wall_time_s")
    for name, rows_, ylabel in (("cost_usd", cost_rows, "estimated cost (USD)"),
                                ("wall_time_s", wall_rows, "wall time (s)")):
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(rows_) + 1)))
        style_ax(ax)
        labels = [r[0] for r in rows_]
        vals = [r[1] for r in rows_]
        ypos = np.arange(len(labels))[::-1]
        ax.barh(ypos, vals, color=BLUE, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(ylabel)
        ax.set_title(f"{ylabel} by run")
        for yi, v in zip(ypos, vals):
            ax.text(v, yi, f" {v:g}", va="center", ha="left", fontsize=9, color=INK2)
        save(fig, os.path.join(cdir, f"{name}.png"))
        meta["charts"][name] = {"title": f"{ylabel} by run", "rows": [{"run": r[0], "value": r[1]} for r in rows_]}

    write_json(os.path.join(cdir, "summary.json"), meta)
    return meta


def collect_scalar(src, key):
    """Collect (label, value) for a scalar across all final_results/manifest files."""
    rows = []
    for rel in [
        "fireworks/metrics/final_results.json",
        "sagemaker/metrics/final_results.json",
        "prime-intellect/metrics/final_results.json",
        "tinker/metrics/final_results.json",
        "verl/metrics/final_results.json",
        "skyrl/metrics/final_results.json",
    ]:
        d = load_json(os.path.join(src, rel))
        for exp, e in d["experiments"].items():
            if key in e and e[key] is not None:
                rows.append((f"{d.get('platform', rel.split('/')[0])}/{exp}", e[key]))
    # miles (flat files)
    for rel, exp in [("miles/metrics/orena_rl_final_results.json", "orena_rl"),
                     ("miles/metrics/cholect50_rl_final_results.json", "cholect50_rl")]:
        d = load_json(os.path.join(src, rel))
        if key in d and d[key] is not None:
            rows.append((f"miles/{exp}", d[key]))
    # miles run_manifest records cost where the flat results file does not
    man = load_json(os.path.join(src, "miles", "metrics", "run_manifest.json"))
    for exp, e in man.get("experiments", {}).items():
        if key in e and e[key] is not None:
            rows.append((f"miles/{exp}", e[key]))
    if key == "estimated_cost_usd":
        rows = [r for r in rows if r[1] is not None]
    # de-duplicate (manifest + flat file may both carry the value)
    seen, dedup = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            dedup.append(r)
    return sorted(dedup, key=lambda r: r[1])


# --------------------------------------------------------------------------- #
# Raw copy + master index
# --------------------------------------------------------------------------- #
RAW_FILES = [
    "fireworks/metrics/final_results.json", "fireworks/metrics/run_manifest.json",
    "fireworks/metrics/orena_sft_metrics.jsonl", "fireworks/metrics/goemotions_sft_metrics.jsonl",
    "sagemaker/metrics/final_results.json", "sagemaker/metrics/run_manifest.json",
    "sagemaker/logs/orena_sft.log", "sagemaker/logs/goemotions_sft.log",
    "sagemaker/logs/orena_rl.log", "sagemaker/logs/goemotions_rl.log",
    "prime-intellect/metrics/final_results.json", "prime-intellect/metrics/run_manifest.json",
    "prime-intellect/logs/exp3_loss.csv",
    "miles/metrics/final_results.json", "miles/metrics/run_manifest.json",
    "miles/metrics/orena_rl_final_results.json", "miles/metrics/cholect50_rl_final_results.json",
    "tinker/metrics/final_results.json", "tinker/metrics/run_manifest.json",
    "tinker/metrics/sft_metrics.jsonl", "tinker/metrics/rl_metrics.jsonl",
    "tinker/metrics/eval_base.json", "tinker/metrics/eval_sft.json", "tinker/metrics/eval_rl.json",
    "verl/metrics/final_results.json", "verl/metrics/run_manifest.json",
    "skyrl/metrics/final_results.json", "skyrl/metrics/run_manifest.json",
    "trainium2/metrics/final_results_smoke.json", "trainium2/metrics/final_results_full_epoch.json",
    "trainium2/metrics/run_manifest.json", "trainium2/logs/run_grpo.log",
    "trainium2/logs/run_grpo_1792850149.log",
    "ds1000/TPU_RL_RESULTS.md", "titanrl/ds1000/ds1000_train.log",
    "skyrl-tpu/README.md", "skyrl-tpu/skyrl_ds1000/README.md",
    "skyrl-tpu/config/gpu_reference_final_results.json",
    "skyrl-tpu/config/gpu_reference_run_manifest.json",
    "skyrl-tpu/skyrl_ds1000/check_results/summary.json",
]


def copy_raw(src, root):
    for rel in RAW_FILES:
        srcp = os.path.join(src, rel)
        if not os.path.exists(srcp):
            continue
        dst = os.path.join(root, "raw", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(srcp, dst)
    print(f"  copied {len(RAW_FILES)} raw files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "ft-rl-platform-benchmark"))
    args = ap.parse_args()
    src = os.path.abspath(args.src)
    root = os.path.dirname(os.path.abspath(__file__))

    all_runs = []
    all_runs += build_fireworks(src, root)
    all_runs += build_sagemaker(src, root)
    all_runs += build_prime_intellect(src, root)
    all_runs += build_miles(src, root)
    all_runs += build_tinker(src, root)
    all_runs += build_verl(src, root)
    all_runs += build_skyrl(src, root)
    all_runs += build_trainium2(src, root)
    all_runs += build_verl_tpu(src, root)
    all_runs += build_titanrl(src, root)
    all_runs += build_skyrl_tpu_cholect50(src, root)
    all_runs += build_skyrl_tpu_ds1000(src, root)
    cross = build_cross_platform(src, root)
    copy_raw(src, root)

    index = {
        "title": "ft-rl-platform-benchmark — generated graphs & metrics",
        "generated_utc": "2026-11-13",
        "runs": [
            {"platform": r["platform"], "experiment": r["experiment"], "task": r["task"],
             "state": r["state"], "graphs": r["graphs"], "series": sorted(r["series"].keys())}
            for r in all_runs
        ],
        "cross_platform_charts": {k: v.get("title") for k, v in cross["charts"].items()},
    }
    write_json(os.path.join(root, "summary.json"), index)
    print(f"\nDone. Root: {root}")


if __name__ == "__main__":
    main()
