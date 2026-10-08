# Language-Balanced Depth Pruning of Small Multilingual LLMs

Research code and released experiment artifacts for studying whether transformer
layer redundancy is language-dependent in small multilingual language models.

The experiments compare English, Hindi, Marathi, and mixed-language calibration
for Qwen, Llama, and Gemma models. They measure layer influence, pruning
stability, perplexity, tokenizer behavior, calibration-budget effects, and
downstream performance on Belebele.

## Research questions

1. Do layer-importance rankings differ between English and Indic text?
2. How does English-calibrated pruning affect Indic-language perplexity?
3. Can a small multilingual calibration mixture preserve performance across
   languages?

## Repository structure

| Path | Purpose |
| --- | --- |
| [`code/`](code/) | Experiment notebooks and execution utilities |
| [`code/executed_runs/`](code/executed_runs/) | Executed v3 notebooks for each model |
| [`data/calibration-cache/`](data/calibration-cache/) | Exact Wikipedia article samples and Belebele subsets used in the reported runs |
| [`docs/research-guide.md`](docs/research-guide.md) | Research protocol and experiment design |
| [`results/`](results/) | Per-model metrics, pruning sets, figures, and configuration files |

The versioned notebooks are retained for provenance:

- `v3_final_language_aware_layer_pruning.ipynb` is the final experiment.
- `v2_corrected_language_aware_layer_pruning.ipynb` is the corrected
  intermediate version.
- `v1_original_language_aware_layer_pruning.ipynb` is the original pilot and
  is not suitable for reproducing the reported results.

## Reproducing the experiments

Install the Python dependencies:

```bash
pip install torch transformers datasets accelerate scipy pandas matplotlib nbclient nbformat ipykernel
```

Run from the repository root:

```bash
cd code
MODEL_KEY=qwen  python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_qwen.ipynb .
MODEL_KEY=llama python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_llama.ipynb .
MODEL_KEY=gemma python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_gemma.ipynb .
```

Each run writes its outputs to `results/<model>/`. The notebooks use the
versioned cache in `data/calibration-cache/`; cached data is not regenerated
unless it is missing.

To regenerate derived figures and tables:

```bash
python code/tools/make_paper_assets.py
```

The reported runs used Python 3.12, PyTorch 2.5.1+cu121, Transformers 4.56.2,
and bfloat16. Model weights are downloaded from Hugging Face on first use and
are not stored in this repository.

## Scope and provenance

This repository contains the computational research artifacts only. The
`results/` directory contains the outputs currently associated with the
reported experiments; the notebooks and cached inputs provide the execution
history needed to audit them.
