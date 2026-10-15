# v4 results

Canonical result bundle for the paper. Every number in the paper can be
recomputed from these files; `python code/tools/audit_claims.py` checks the
central claims against them.

| Directory | Checkpoint | Revision |
| --- | --- | --- |
| `qwen/` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `llama/` | `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| `gemma/` | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` |

`manifest.json` lists SHA-256 digests and sizes for the files used in the
paper (digests are computed on LF line endings, as stored in git).

## Files in each model directory

| File | Contents |
| --- | --- |
| `experiment_config.json` | Model id, immutable revision, software environment, hyperparameters, evaluation fingerprint |
| `data_split.csv`, `calibration_article_ids.csv`, `evaluation_article_ids.csv` | Article-disjoint calibration/evaluation splits (identical across models) |
| `baseline_metrics.csv` | Unpruned perplexity and bits per character per language |
| `tokenizer_tokens_per_word.csv` | Tokenizer fertility per language |
| `block_influence_bi_by_language.csv` | Per-block BI for EN/HI/MR/TA |
| `block_influence_rl2_by_language.csv` | Per-block relative-L2 diagnostic |
| `block_influence_by_language.csv` | Per-block BI for each language and the three balanced aggregators |
| `late_layer_shift_summary.csv`, `layer_bi_shift_bootstrap.csv` | Late-vs-middle BI contrasts with bootstrap CIs |
| `rank_agreement.csv`, `ranking_stability.csv` | Spearman/Jaccard agreement across languages and calibration seeds |
| `pruning_sets.json`, `primary_pruning_sets.json` | Full-pool and per-seed selected blocks |
| `primary_pruning_results.csv` | 420 primary rows: 7 selectors x 5 seeds x 3 depths x 4 languages |
| `primary_bootstrap_contrasts.csv` | 99 hierarchical-bootstrap contrasts (EN vs. balanced, aggregators) |
| `strategy_summary.csv` | Mean/worst target-language degradation per selector |
| `random_pruning_results.csv` | Ten random block sets per depth |
| `reference_pruning_results.csv` | Full-pool reference selections (descriptive) |
| `diagnostic_v1_last_layer.csv` | Last-block BI with and without the final-norm bug fixed in v2 (diagnostic) |
| `fig1`-`fig4` PNGs | Per-model diagnostic plots produced by the notebook |

Model weights, downloaded Wikipedia text, and per-chunk NLL / influence caches
(`*.npz`) are not redistributed.
