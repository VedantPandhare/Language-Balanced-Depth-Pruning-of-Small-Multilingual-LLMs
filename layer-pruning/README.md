# Language-dependent layer redundancy in small multi-lingual LLMs

> **Working title:** Do Small LLMs Share Redundant Layers Across Languages? Language-Aware Layer Pruning for English, Hindi, and Marathi

## About the Project

Modern transformer-based LLMs can be made faster and smaller by **layer pruning** — identifying and permanently removing transformer layers that contribute little to the model's output. The standard recipe for finding these "unimportant" layers relies on calibration text written in **English**.

This project asks a simple but overlooked question:

> *Are the redundant layers the same for Hindi, Marathi, and code-mixed text as they are for English? If not, pruning calibrated on English silently damages Indic-language performance.*

### The Problem

Layer-pruning methods score each transformer layer by measuring how much it changes its own input (e.g. via cosine similarity or block influence). Layers with near-zero influence are then dropped. Because this calibration step almost always uses English Wikipedia or similar English corpora, there is an implicit assumption that layer importance is **language-agnostic**. For monolingual English models that may be fine; for small multilingual LLMs deployed in linguistically diverse settings — such as Hindi or Marathi — the assumption has never been rigorously tested.

### Why it Matters

- **Indic languages are structurally different** from English (richer morphology, different script, SOV word order), so the model may rely on different layers to process them.
- **Small LLMs** (1–3 B parameters) are the realistic deployment target for low-resource, on-device applications in India and similar markets.
- If layer importance is language-dependent, then standard English-calibrated pruning quietly erodes multilingual capability — a failure mode that is invisible if you only evaluate on English benchmarks.
- The fix (language-matched or mixed calibration) is cheap; the risk of not checking is real.

### Research Questions

| # | Question |
|---|----------|
| **RQ1** | Do layer-importance rankings differ between English and Indic text? |
| **RQ2** | How much worse is Indic perplexity when pruning with English calibration vs. language-matched calibration? |
| **RQ3** | Does a small multilingual calibration mix fix the problem, and how many samples does it need? |

### Models Studied

Three small open-weight multilingual LLMs are evaluated: **Qwen**, **LLaMA**, and **Gemma** (1–3 B parameter range). Each is probed with Wikipedia samples in **English**, **Hindi**, and **Marathi** (1,500 samples per language, seed 42) and evaluated on downstream comprehension using the **Belebele** benchmark.

---

## Folder layout

| Folder | Contents |
|---|---|
| `paper/` | The LaTeX paper (`main.tex`, `references.bib`, `IEEEtran.cls`), `figures/` (vector PDFs), `tables/` (auto-generated `.tex` tables plus the CSV/JSON numbers behind them) and the compiled `main.pdf` |
| `results/` | Raw outputs for each model (`qwen/`, `llama/`, `gemma/`): per-layer influence, pruning sets, perplexity sweeps, random and deep-block controls, bootstrap contrasts, seeds, budget sweep, Belebele, per-chunk NLLs (`.npz`), figures and run config |
| `code/` | All code and data |

### What is in `code/`

- **`v3_final_language_aware_layer_pruning.ipynb`**: the final experiment notebook. It produced every number in the paper.
- **`executed_runs/`**: the three executed copies of the v3 notebook (Qwen, Llama, Gemma), with all outputs.
- **`v2_corrected_language_aware_layer_pruning.ipynb`**: the intermediate corrected version, kept for history.
- **`v1_original_language_aware_layer_pruning.ipynb`**: the original pilot notebook. It contains known bugs, documented in the v2/v3 header table. Do not use it for results.
- **`tools/build_notebook.py`**: generates the v3 notebook.
- **`tools/run_nb.py`**: headless notebook runner.
- **`tools/make_paper_assets.py`**: builds every paper figure and table from `results/`.
- **`data_cache/`**: the exact Wikipedia article samples (seed 42, 1,500 per language) and the Belebele subsets that were used.
- **`language_aware_layer_pruning_IEEE_research_guide.md`**: the original research guide.

## Reproduce

```bash
pip install torch transformers datasets accelerate scipy pandas matplotlib nbclient nbformat ipykernel
cd code
MODEL_KEY=qwen  python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_qwen.ipynb .
MODEL_KEY=llama python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_llama.ipynb .
MODEL_KEY=gemma python tools/run_nb.py v3_final_language_aware_layer_pruning.ipynb executed_runs/v3_gemma.ipynb .
cd ..
python code/tools/make_paper_assets.py        # regenerate paper/figures and paper/tables
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Each model run takes about 20–45 minutes on an 8 GB RTX 4060. Models download from Hugging Face on the first run; the model weights were deleted after the experiments to free disk space.

By default Llama and Gemma load from the ungated `unsloth/` mirrors. To use the official checkpoints, set `LLAMA_ID=meta-llama/Llama-3.2-1B` and `GEMMA_ID=google/gemma-2-2b` after `huggingface-cli login`.

Environment used for the reported runs: Python 3.12, PyTorch 2.5.1+cu121, Transformers 4.56.2, bfloat16.
