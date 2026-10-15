"""Fail fast if the paper's central quantitative claims drift from the v4 results.

Usage (from the repository root):  python code/tools/audit_claims.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = {
    "qwen": ROOT / "results" / "qwen",
    "llama": ROOT / "results" / "llama",
    "gemma": ROOT / "results" / "gemma",
}


def read(model: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS[model] / name)


expected_ids = {
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "llama": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "gemma": ("google/gemma-2-2b", "c5ebcd40d208330abc697524c919956e692655cf"),
}

for model, root in RESULTS.items():
    config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    assert (config["model_id"], config["model_revision"]) == expected_ids[model]
    primary = read(model, "primary_pruning_results.csv")
    contrasts = read(model, "primary_bootstrap_contrasts.csv")
    assert len(primary) == 420 and len(contrasts) == 99
    assert not primary.isna().any().any()
    assert not contrasts[["delta_logppl", "ci_low", "ci_high", "ppl_ratio"]].isna().any().any()

reference_eval = read("qwen", "evaluation_article_ids.csv")
reference_calib = read("qwen", "calibration_article_ids.csv")
for model in ("llama", "gemma"):
    assert reference_eval.equals(read(model, "evaluation_article_ids.csv"))
    assert reference_calib.equals(read(model, "calibration_article_ids.csv"))

late = pd.concat([read(model, "late_layer_shift_summary.csv") for model in RESULTS])
assert len(late) == 9
assert (late.contrast_ci_low > 0).all()
assert np.isclose(late.late_geometric_shift_pct.min(), 20.7188212337)
assert np.isclose(late.late_geometric_shift_pct.max(), 55.5467516255)

def en_vs_raw(model: str) -> pd.DataFrame:
    frame = read(model, "primary_bootstrap_contrasts.csv")
    return frame[(frame.contrast_family == "en_vs_lb") & (frame.condition_b == "lb_mean")]

qwen_k6 = en_vs_raw("qwen").query("k == 6").set_index("eval_language")
assert (qwen_k6.loc[["hi", "mr", "ta"], "ci_low"] > 0).all()

gemma = en_vs_raw("gemma")
assert (gemma.ci_low > 0).all()

llama_k2 = en_vs_raw("llama").query("k == 2").set_index("eval_language")
assert llama_k2.loc["hi", "ppl_ratio"] > 1
assert llama_k2.loc["mr", "ppl_ratio"] > 1
assert llama_k2.loc["ta", "ppl_ratio"] < 1
assert np.isclose(1 / llama_k2.loc["ta", "ppl_ratio"], 17.478722, rtol=1e-5)

print("Claim audit passed: provenance, row counts, splits, late contrasts, and key PPL ratios.")
