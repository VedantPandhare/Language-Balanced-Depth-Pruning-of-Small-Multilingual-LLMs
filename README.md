# Language-Balanced Depth Pruning of Small Multilingual LLMs

Research code and experiment artifacts for studying calibration-language
sensitivity in Block Influence (BI) pruning of small multilingual language
models.

The current v4 study tests whether English-only calibration produces less
robust pruning decisions for Hindi and Marathi, and whether equal-budget
language-balanced calibration improves block selection. Tamil is included as
a held-out language from a different Indic script and language family.

## Research questions

1. Do BI rankings and selected pruning sets differ across English, Hindi,
   Marathi, and Tamil?
2. Does English-only calibration transfer worse than matched-language or
   language-balanced calibration?
3. Which equal-budget language-balanced aggregator (raw mean, rank mean, or
   minimax rank) is most robust, including on held-out Tamil?

## Experimental scope

- Models: Qwen2.5-1.5B-Instruct, Llama-3.2-1B, and Gemma-2-2B.
- Calibration languages: English, Hindi, and Marathi.
- Held-out control language: Tamil; it is evaluated but is not included in the
  language-balanced aggregators.
- Pruning levels: 2, 4, and 6 transformer blocks.
- Primary calibration budget: 150 chunks for every selector.
- Aggregators: raw BI mean, normalized rank mean, and minimax normalized rank.
- Primary outcomes: pruning-set stability and per-language perplexity changes.

The current study does not use healing or post-pruning fine-tuning. Belebele
and the calibration-budget sweep remain optional appendix experiments and are
disabled in the primary run.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`code/`](code/) | Experiment notebooks and execution utilities |
| [`code/v4_robust_language_balanced_block_pruning.ipynb`](code/v4_robust_language_balanced_block_pruning.ipynb) | Current Colab/A100 experiment |
| [`code/executed_runs/`](code/executed_runs/) | Executed notebooks retained for provenance |
| [`data/calibration-cache/`](data/calibration-cache/) | Deterministic calibration and evaluation samples |
| [`docs/research-guide.md`](docs/research-guide.md) | Original research protocol |
| [`results/`](results/) | Archived v3 experiment outputs |
| `results_v4/` | Restartable outputs created by v4 runs |
| `paper_v4_assets/` | Combined v4 tables, manifests, and figures |

The v1-v3 notebooks are retained for provenance. The v4 notebook is the source
for the revised claims and new results.

## Run the v4 experiments

Install dependencies in a fresh Colab A100 runtime:

```bash
pip install -q "torch>=2.5" "transformers>=4.56" datasets accelerate sentencepiece scipy pandas matplotlib nbclient nbformat ipykernel
hf auth login
```

Llama and Gemma require access to their official gated Hugging Face
repositories. Run each model sequentially from the repository root:

```bash
cd code

RUN_REFERENCE_SWEEP=1 RUN_BUDGET_SWEEP=0 RUN_BELEBELE=0 MODEL_KEY=qwen \
  python tools/run_nb.py v4_robust_language_balanced_block_pruning.ipynb executed_runs/v4_qwen.ipynb .

RUN_REFERENCE_SWEEP=1 RUN_BUDGET_SWEEP=0 RUN_BELEBELE=0 MODEL_KEY=llama \
  python tools/run_nb.py v4_robust_language_balanced_block_pruning.ipynb executed_runs/v4_llama.ipynb .

RUN_REFERENCE_SWEEP=1 RUN_BUDGET_SWEEP=0 RUN_BELEBELE=0 MODEL_KEY=gemma \
  python tools/run_nb.py v4_robust_language_balanced_block_pruning.ipynb executed_runs/v4_gemma.ipynb .

cd ..
python code/tools/make_v4_paper_assets.py
```

Set `RESULTS_ROOT` and `DATA_CACHE_DIR` to persistent Google Drive paths when
running on Colab. The notebook caches influence arrays and each evaluated
pruning configuration, so repeating an interrupted model run resumes completed
work. `PAPER_ASSETS_DIR` optionally changes the asset builder's output path.

## Reproduce the archived v3 experiments

```bash
cd code
MODEL_KEY=qwen  python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_qwen.ipynb .
MODEL_KEY=llama python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_llama.ipynb .
MODEL_KEY=gemma python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_gemma.ipynb .
cd ..
python code/tools/make_paper_assets.py
```

The archived v3 notebook used `unsloth/` mirrors by default. The v4 notebook
uses the official model repositories. Model weights are downloaded from
Hugging Face and are not stored in this repository.

## Scope and provenance

This repository contains computational research artifacts. The paper PDF and
LaTeX manuscript are intentionally maintained outside version control. Claims
about Tamil, late-layer concentration, and the strongest aggregation rule
should be finalized only after all three v4 model runs and confidence-interval
analyses complete.
