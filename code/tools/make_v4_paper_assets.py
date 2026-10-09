"""Build paper-ready v4 tables and figures after all A100 runs finish."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("RESULTS_ROOT", ROOT / "results_v4"))
OUTPUT = Path(os.environ.get("PAPER_ASSETS_DIR", ROOT / "paper_v4_assets"))
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
for path in (FIGURES, TABLES):
    path.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen", "llama", "gemma"]
MODEL_LABELS = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-1B", "gemma": "Gemma-2-2B"}
LANGS = ["en", "hi", "mr", "ta"]
CONDITIONS = ["en", "hi", "mr", "ta", "lb_mean", "lb_rank_mean", "lb_minimax_rank"]
PRIMARY = ["en", "lb_mean", "lb_rank_mean", "lb_minimax_rank"]
LABELS = {
    "en": "EN-only",
    "hi": "HI-only",
    "mr": "MR-only",
    "ta": "TA-only",
    "lb_mean": "LB mean",
    "lb_rank_mean": "LB rank-mean",
    "lb_minimax_rank": "LB minimax-rank",
}
COLORS = {
    "en": "#2a78d6",
    "hi": "#eb6834",
    "mr": "#1baf7a",
    "ta": "#c13d8a",
    "lb_mean": "#eda100",
    "lb_rank_mean": "#159f9a",
    "lb_minimax_rank": "#7b51b8",
}


def read_result(model: str, filename: str) -> pd.DataFrame:
    path = RESULTS / model / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Complete the {model} v4 notebook run first.")
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame["model"] = model
        frame = frame[["model"] + [c for c in frame.columns if c != "model"]]
    else:
        frame.insert(0, "model", model)
    return frame


def combine(filename: str) -> pd.DataFrame:
    return pd.concat([read_result(model, filename) for model in MODELS], ignore_index=True)


def write_latex(frame: pd.DataFrame, filename: str, caption: str, label: str) -> None:
    text = frame.to_latex(index=False, escape=True, caption=caption, label=label, float_format="%.3f")
    (TABLES / filename).write_text(text, encoding="utf-8")


configs = {}
for model in MODELS:
    path = RESULTS / model / "experiment_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Complete all three model runs before building assets.")
    configs[model] = json.loads(path.read_text(encoding="utf-8"))

# Cross-model comparisons must use the same source articles, not merely the same dataset seed.
for split_file in ["evaluation_article_ids.csv", "calibration_article_ids.csv"]:
    reference = pd.read_csv(RESULTS / MODELS[0] / split_file).sort_values(["language", "chunk_index"]).reset_index(drop=True)
    for model in MODELS[1:]:
        candidate = pd.read_csv(RESULTS / model / split_file).sort_values(["language", "chunk_index"]).reset_index(drop=True)
        if not reference.equals(candidate):
            raise ValueError(f"Article split mismatch between {MODELS[0]} and {model}: {split_file}")

primary = combine("primary_pruning_results.csv")
strategy = combine("strategy_summary.csv")
contrasts = combine("primary_bootstrap_contrasts.csv")
late = combine("late_layer_shift_summary.csv")
stability = combine("ranking_stability.csv")

primary.to_csv(TABLES / "all_primary_pruning_results.csv", index=False)
strategy.to_csv(TABLES / "all_strategy_summaries.csv", index=False)
contrasts.to_csv(TABLES / "all_primary_contrasts.csv", index=False)
late.to_csv(TABLES / "all_late_layer_shifts.csv", index=False)
stability.to_csv(TABLES / "all_ranking_stability.csv", index=False)
(TABLES / "run_manifest.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

# Compact method comparison at the middle pruning level.
method_table = strategy[(strategy.pruned_layers == 4) & strategy.calibration.isin(PRIMARY)].copy()
method_table["model"] = method_table.model.map(MODEL_LABELS)
method_table["selector"] = method_table.calibration.map(LABELS)
method_table = method_table[["model", "selector", "target_mean_mean", "target_mean_std",
                             "target_worst_mean", "target_worst_std"]]
write_latex(
    method_table,
    "aggregator_ablation_k4.tex",
    "Equal-budget BI selector comparison at four removed blocks. Values are relative perplexity increases in percent over five calibration seeds.",
    "tab:aggregator-ablation",
)

# English versus each LB selector, including held-out Tamil.
transfer_table = contrasts[
    (contrasts.k == 4)
    & (contrasts.contrast_family == "en_vs_lb")
    & contrasts.condition_b.isin(["lb_mean", "lb_rank_mean", "lb_minimax_rank"])
].copy()
transfer_table["model"] = transfer_table.model.map(MODEL_LABELS)
transfer_table["LB selector"] = transfer_table.condition_b.map(LABELS)
transfer_table["language"] = transfer_table.eval_language.str.upper()
transfer_table["ratio_ci_low"] = np.exp(transfer_table.ci_low)
transfer_table["ratio_ci_high"] = np.exp(transfer_table.ci_high)
transfer_table = transfer_table[["model", "language", "LB selector", "ppl_ratio",
                                 "ratio_ci_low", "ratio_ci_high", "p_holm"]]
write_latex(
    transfer_table,
    "english_vs_lb_k4.tex",
    "English-only versus language-balanced calibration at four removed blocks. Ratios above one favor LB calibration.",
    "tab:english-vs-lb",
)

late_table = late.copy()
late_table["model"] = late_table.model.map(MODEL_LABELS)
late_table["language"] = late_table.target_language.str.upper()
late_table = late_table[["model", "language", "late_geometric_shift_pct", "late_minus_middle_logratio",
                         "contrast_ci_low", "contrast_ci_high", "late_stronger_than_middle"]]
write_latex(
    late_table,
    "late_layer_shift.tex",
    "Target-language BI shift in the predeclared late-layer segment relative to English.",
    "tab:late-layer-shift",
)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.2, "figure.dpi": 120, "savefig.dpi": 220})

# Figure 1: raw per-language BI profiles. Aggregator scores are intentionally excluded because rank
# aggregators have a different numerical scale.
fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.2), sharey=False)
for ax, model in zip(axes, MODELS):
    frame = pd.read_csv(RESULTS / model / "block_influence_bi_by_language.csv")
    for lang in LANGS:
        ax.plot(frame.layer, frame[lang], marker="o", ms=3, color=COLORS[lang], label=lang.upper())
    ax.set_yscale("log")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Block Influence")
    ax.set_title(MODEL_LABELS[model], loc="left")
axes[0].legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "language_bi_profiles.pdf")
plt.close(fig)

# Figure 2: primary average and worst-target degradation across pruning levels.
fig, axes = plt.subplots(2, len(MODELS), figsize=(15, 7.2), sharex=True)
for col, model in enumerate(MODELS):
    subset = strategy[strategy.model == model]
    for condition in PRIMARY:
        rows = subset[subset.calibration == condition].sort_values("pruned_layers")
        axes[0, col].errorbar(rows.pruned_layers, rows.target_mean_mean, yerr=rows.target_mean_std,
                              marker="o", capsize=3, color=COLORS[condition], label=LABELS[condition])
        axes[1, col].errorbar(rows.pruned_layers, rows.target_worst_mean, yerr=rows.target_worst_std,
                              marker="o", capsize=3, color=COLORS[condition], label=LABELS[condition])
    axes[0, col].set_title(MODEL_LABELS[model], loc="left")
    axes[0, col].set_yscale("symlog", linthresh=1)
    axes[1, col].set_yscale("symlog", linthresh=1)
    axes[1, col].set_xlabel("Blocks removed")
axes[0, 0].set_ylabel("Mean target PPL increase (%)")
axes[1, 0].set_ylabel("Worst target PPL increase (%)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "aggregator_mean_worst_degradation.pdf")
plt.close(fig)

# Figure 3: late-versus-middle log-BI contrast with 95% intervals.
fig, axes = plt.subplots(1, len(MODELS), figsize=(13.5, 3.8), sharey=True)
for ax, model in zip(axes, MODELS):
    rows = late[late.model == model].set_index("target_language").loc[["hi", "mr", "ta"]]
    values = rows.late_minus_middle_logratio.to_numpy()
    lower = values - rows.contrast_ci_low.to_numpy()
    upper = rows.contrast_ci_high.to_numpy() - values
    ax.errorbar(np.arange(3), values, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color="#202020")
    ax.axhline(0, color="#777777", lw=1)
    ax.set_xticks(np.arange(3), ["HI", "MR", "TA"])
    ax.set_title(MODEL_LABELS[model], loc="left")
    ax.set_xlabel("Target language")
axes[0].set_ylabel("Late minus middle log-BI ratio")
fig.tight_layout()
fig.savefig(FIGURES / "late_layer_contrast.pdf")
plt.close(fig)

print(f"Wrote v4 paper assets to {OUTPUT}")
