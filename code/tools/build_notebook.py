"""Build the v4 Colab/A100 notebook for robust multilingual block pruning."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Robust Language-Balanced Block-Influence Pruning - v4 (Colab/A100)

**Research question:** Does English-only calibration select less robust blocks for Hindi and Marathi,
especially in later layers, and do language-balanced selectors transfer to held-out Tamil?

Primary calibration comparisons use the same total number of tokens. Language-balanced selectors are
computed over English, Hindi, and Marathi; Tamil is a held-out cross-script Indic control. The notebook is
restartable: influence arrays and every evaluated pruning configuration are cached under `results/`.

## What changed vs. v1 (and why it matters for the paper)

| # | v1 problem | Effect on results | v2 fix |
|---|---|---|---|
| 1 | Calibration and evaluation chunks were built from the **same first rows** of the same stream → identical data | Calibration/evaluation leakage; violates the guide's own rule | Article-level disjoint split, asserted |
| 2 | Last layer's "output" was `hidden_states[-1]`, which in HF already has the **final RMSNorm** applied | Last-layer influence is wrong (compares un-normed input with normed output) | Forward hooks capture true block outputs |
| 3 | Pruning loop runs `del model` on every iteration → `NameError` on iteration 2; later cells also use the deleted `model` | Notebook cannot run top-to-bottom | Layers are swapped out and restored on one model (weights untouched; equivalent to a fresh model, verified) |
| 4 | Stability = shuffle the **same 20 chunks** twice; influence is a mean, so order is irrelevant → Jaccard ≡ 1.0 | Stability metric is meaningless | Independent random calibration subsets across seeds (mean ± std) |
| 5 | Random baseline only printed layer ids, never evaluated | No control | Random (10 seeds) and a "deep contiguous block" heuristic are evaluated |
| 6 | First rows of the stream, 20 chunks, single seed | Tiny, biased sample | Shuffled articles, 200 calibration and 200 evaluation articles per language, 5 seeds |
| 7 | Mixed calibration silently used three times the data | Token-budget confound | Every primary selector uses the same 150-chunk total budget |
| 8 | No uncertainty on PPL differences | Claims not testable | Hierarchical CIs over calibration seeds and independent evaluation articles |
| 9 | Chunks had no BOS token | Gemma-2 baseline PPL explodes (EN ≈182, MR ≈2557); Llama slightly off | BOS prepended for models that define one; BOS position excluded from influence |
| 10 | `json.dumps` on numpy int64 (cell 24) | v1 crashes before pruning starts | Layer ids cast to `int` |

Extras added from the research guide: relative-L2 influence metric, calibration-budget sweep, cross-language
Jaccard of pruning sets, tokenizer statistics on the same articles, multi-model support.

Set `MODEL_KEY` to `qwen`, `llama`, or `gemma`. Official model repositories are used by default.
The compute-heavy cells are tuned for an A100 but expose batch-size environment variables.
""")

code(r"""
# Fresh Colab runtimes only: uncomment and run once, then restart the runtime if requested.
# %pip install -q "torch>=2.5" "transformers>=4.56" datasets accelerate sentencepiece scipy pandas matplotlib nbformat
""")

code(r"""
import os, gc, math, json, random, hashlib, warnings, itertools, time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.stats import spearmanr, kendalltau
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_KEY = os.environ.get("MODEL_KEY", "qwen")
MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": os.environ.get("LLAMA_ID", "meta-llama/Llama-3.2-1B"),
    "gemma": os.environ.get("GEMMA_ID", "google/gemma-2-2b"),
}
LANGS = ["en", "hi", "mr", "ta"]
AGG_LANGS = ["en", "hi", "mr"]
TARGET_LANGS = ["hi", "mr", "ta"]
LANG_CODES = {"en": "20231101.en", "hi": "20231101.hi", "mr": "20231101.mr", "ta": "20231101.ta"}
SINGLE_CONDITIONS = list(LANGS)
AGG_CONDITIONS = ["lb_mean", "lb_rank_mean", "lb_minimax_rank"]
CONDITIONS = SINGLE_CONDITIONS + AGG_CONDITIONS

SEQ_LEN = 256
CALIB_POOL_CHUNKS = 200        # one chunk from each of 200 calibration articles per language
EVAL_CHUNKS = 200              # one chunk from each of 200 held-out articles per language
MAX_CHUNKS_PER_ARTICLE = 1     # makes the statistical unit an independent article
N_ARTICLES_TO_SCAN = 1000
SHUFFLE_BUFFER = 5000
MIN_WORDS = 300                 # ensures every tokenizer can form a full 256-token chunk

PRUNE_LEVELS = [2, 4, 6]
N_SEEDS = 5                    # calibration-subset seeds
PRIMARY_BUDGET = 150           # equal total budget: LB selectors use 50 chunks per aggregation language
BUDGETS = [15, 30, 60, 90, 150]
BUDGET_K = 4
N_RANDOM_SEEDS = 10
N_BOOT = 5000
_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
_a100_scale = _gpu_mem_gb >= 35
INF_BATCH = int(os.environ.get("INF_BATCH", 32 if _a100_scale else (8 if MODEL_KEY == "gemma" else 16)))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", 32 if _a100_scale else (4 if MODEL_KEY == "gemma" else 8)))
RUN_REFERENCE_SWEEP = os.environ.get("RUN_REFERENCE_SWEEP", "1") == "1"
RUN_BUDGET_SWEEP = os.environ.get("RUN_BUDGET_SWEEP", "0") == "1"
RUN_BELEBELE = os.environ.get("RUN_BELEBELE", "0") == "1"
N_BELEBELE = 300
BELEBELE_RANDOM_SEEDS = 2           # random-pruning controls evaluated on Belebele
BELEBELE_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "ta": "tam_Taml"}

ROOT = Path.cwd()                                   # run from the code/ folder
_default_data_cache = ROOT.parent / "data" / "calibration-cache" if ROOT.name == "code" else ROOT / "data" / "calibration-cache"
DATA_CACHE = Path(os.environ.get("DATA_CACHE_DIR", _default_data_cache)); DATA_CACHE.mkdir(parents=True, exist_ok=True)
_default_results = ROOT.parent / "results" if ROOT.name == "code" else ROOT / "results"
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", _default_results)); RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = RESULTS_ROOT / MODEL_KEY; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NLL_CACHE_DIR = OUTPUT_DIR / "nll_cache"; NLL_CACHE_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else (
    torch.float16 if DEVICE == "cuda" else torch.float32)

# Fixed colour per condition across every figure (identity, never rank)
COLORS = {"en": "#2a78d6", "hi": "#eb6834", "mr": "#1baf7a", "ta": "#c13d8a",
          "lb_mean": "#eda100", "lb_rank_mean": "#159f9a", "lb_minimax_rank": "#7b51b8",
          "random": "#8a8984", "deep_block": "#4a3aa7"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "lines.linewidth": 2,
                     "figure.dpi": 110, "savefig.dpi": 200})

print("torch", torch.__version__, "| transformers", transformers.__version__)
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "| dtype:", DTYPE)
print("Model:", MODELS[MODEL_KEY], "| output:", OUTPUT_DIR)
print("Batches: influence", INF_BATCH, "evaluation", EVAL_BATCH, "| primary budget", PRIMARY_BUDGET)
""")

md("## 1. Load model (once) and locate Transformer blocks")
code(r"""
def load_model(model_key):
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=DTYPE)
    if "gemma" in name.lower():
        kwargs["attn_implementation"] = "eager"   # recommended for Gemma-2 soft-capping
    mdl = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(DEVICE)
    mdl.eval()
    return tok, mdl

BLOCK_PATHS = ["model.layers", "model.decoder.layers", "transformer.h", "decoder.layers"]

def _resolve(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj

def get_transformer_blocks(model):
    for path in BLOCK_PATHS:
        try:
            blocks = _resolve(model, path)
            if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                return blocks, path
        except AttributeError:
            pass
    raise RuntimeError("Could not find Transformer blocks.")

def set_transformer_blocks(model, new_blocks, path):
    parent_path, attr = path.rsplit(".", 1)
    setattr(_resolve(model, parent_path), attr, nn.ModuleList(new_blocks))

tokenizer, model = load_model(MODEL_KEY)
BLOCKS, BLOCK_PATH = get_transformer_blocks(model)
ORIGINAL_BLOCKS = list(BLOCKS)
N_LAYERS = len(ORIGINAL_BLOCKS)
cfg = model.config
model_info = {
    "model_key": MODEL_KEY, "model_id": MODELS[MODEL_KEY],
    "model_revision": getattr(cfg, "_commit_hash", None),
    "parameters": sum(p.numel() for p in model.parameters()),
    "layers": N_LAYERS, "hidden_size": cfg.hidden_size,
    "attention_heads": cfg.num_attention_heads,
    "kv_heads": getattr(cfg, "num_key_value_heads", None),
    "vocab_size": len(tokenizer), "block_path": BLOCK_PATH,
}
print(json.dumps(model_info, indent=2))
""")

md(r"""
## 2. Data: shuffled Wikipedia, article-level disjoint calibration / evaluation

Articles are sampled from a shuffled stream and cached to disk, so **every model sees exactly the same
articles** (token chunks differ only because tokenizers differ). Evaluation articles never appear in calibration.
""")
code(r"""
def load_articles(lang):
    cache = DATA_CACHE / f"{lang}_articles_seed{SEED}_n{N_ARTICLES_TO_SCAN}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    ds = load_dataset("wikimedia/wikipedia", LANG_CODES[lang], split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    arts = []
    for row in ds:
        text = row.get("text", "")
        if text and len(text.split()) >= MIN_WORDS:
            arts.append({"id": row["id"], "title": row["title"], "text": text})
        if len(arts) >= N_ARTICLES_TO_SCAN:
            break
    cache.write_text(json.dumps(arts, ensure_ascii=False), encoding="utf-8")
    return arts

# Models with a BOS token (Llama, Gemma) need it: Gemma-2 without <bos> gives PPL in the hundreds/thousands.
# Each chunk = [BOS] + (SEQ_LEN-1) text tokens; Qwen has no BOS, so it gets SEQ_LEN text tokens.
BOS = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
BODY = SEQ_LEN - len(BOS)
SKIP = len(BOS)   # positions excluded from influence (BOS is an attention sink with outlier norms)
print("BOS prepended:", bool(BOS))

def article_chunks(text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    out = []
    for s in range(0, len(ids) - BODY + 1, BODY):
        out.append(BOS + ids[s:s + BODY])
        if len(out) >= MAX_CHUNKS_PER_ARTICLE:
            break
    return out

articles, calib_pool, eval_chunks, split_stats = {}, {}, {}, []
calib_article_ids, eval_article_ids = {}, {}
for lang in LANGS:
    t0 = time.time()
    articles[lang] = load_articles(lang)
    ev, ca, ev_ids, ca_ids = [], [], [], []
    for art in articles[lang]:
        ch = article_chunks(art["text"])
        if not ch:
            continue
        if len(ev) < EVAL_CHUNKS:                       # first valid articles -> evaluation
            ev.append(ch[0]); ev_ids.append(str(art["id"]))
        elif len(ca) < CALIB_POOL_CHUNKS:               # remaining valid articles -> calibration
            ca.append(ch[0]); ca_ids.append(str(art["id"]))
        else:
            break
    assert not (set(ev_ids) & set(ca_ids)), "article overlap between calibration and evaluation"
    assert len(set(ev_ids)) == len(ev_ids) and len(set(ca_ids)) == len(ca_ids), "article IDs are not unique"
    assert len(ev) == EVAL_CHUNKS and len(ca) == CALIB_POOL_CHUNKS, (lang, len(ev), len(ca))
    hashes = lambda xs: {hashlib.md5(str(x).encode()).hexdigest() for x in xs}
    assert not (hashes(ev) & hashes(ca)), "identical chunk in both splits"
    eval_chunks[lang] = torch.tensor(ev, dtype=torch.long)
    calib_pool[lang] = torch.tensor(ca, dtype=torch.long)
    eval_article_ids[lang] = np.asarray(ev_ids)
    calib_article_ids[lang] = np.asarray(ca_ids)
    split_stats.append({"language": lang, "eval_articles": len(set(ev_ids)), "calib_articles": len(set(ca_ids)),
                        "eval_chunks": len(ev), "calib_chunks": len(ca),
                        "eval_tokens": len(ev) * SEQ_LEN, "calib_tokens": len(ca) * SEQ_LEN})
    print(lang, "loaded in %.0fs" % (time.time() - t0))

split_df = pd.DataFrame(split_stats)
split_df.to_csv(OUTPUT_DIR / "data_split.csv", index=False)
pd.concat([pd.DataFrame({"language": l, "chunk_index": np.arange(len(eval_article_ids[l])),
                         "article_id": eval_article_ids[l]}) for l in LANGS]).to_csv(
    OUTPUT_DIR / "evaluation_article_ids.csv", index=False)
pd.concat([pd.DataFrame({"language": l, "chunk_index": np.arange(len(calib_article_ids[l])),
                         "article_id": calib_article_ids[l]}) for l in LANGS]).to_csv(
    OUTPUT_DIR / "calibration_article_ids.csv", index=False)
display(split_df)
""")

md("## 3. Tokenizer statistics (same articles for every model)")
code(r"""
tok_rows = []
for lang in LANGS:
    tpw, cpt = [], []
    for art in articles[lang][:500]:
        words = art["text"].split()
        n_tok = len(tokenizer(art["text"], add_special_tokens=False)["input_ids"])
        tpw.append(n_tok / len(words)); cpt.append(len(art["text"]) / n_tok)
    tok_rows.append({"language": lang, "tokens_per_word_mean": np.mean(tpw),
                     "tokens_per_word_median": np.median(tpw), "chars_per_token_mean": np.mean(cpt),
                     "n_articles": len(tpw)})
token_stats_df = pd.DataFrame(tok_rows)
token_stats_df.to_csv(OUTPUT_DIR / "tokenizer_tokens_per_word.csv", index=False)
display(token_stats_df)
""")

md(r"""
## 4. Block influence with true block inputs/outputs

For block $l$ and token $t$: $BI_l = 1 - \cos(h^{in}_{l,t}, h^{out}_{l,t})$ and the alternative
$RL2_l = \lVert h^{out} - h^{in}\rVert / \lVert h^{in}\rVert$, both captured with hooks on the block itself.

We store a **per-chunk** influence matrix (chunks × layers). Any calibration subset's influence is then the
mean of its rows, so stability and budget sweeps need no extra forward passes.
A diagnostic below quantifies the v1 last-layer bug.
""")
code(r"""
def _first_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x

@torch.no_grad()
def per_chunk_influence(chunks, check_v1=False):
    blocks, _ = get_transformer_blocks(model)
    store = {}
    def pre(i):
        def fn(module, args, kwargs):
            store[("in", i)] = args[0] if args else kwargs["hidden_states"]
        return fn
    def post(i):
        def fn(module, args, kwargs, output):
            store[("out", i)] = _first_tensor(output)
        return fn
    handles = [b.register_forward_pre_hook(pre(i), with_kwargs=True) for i, b in enumerate(blocks)]
    handles += [b.register_forward_hook(post(i), with_kwargs=True) for i, b in enumerate(blocks)]
    cos_all, l2_all, v1_gap = [], [], []
    try:
        for s in range(0, len(chunks), INF_BATCH):
            ids = chunks[s:s + INF_BATCH].to(DEVICE)
            store.clear()
            out = model.model(input_ids=ids, use_cache=False, output_hidden_states=check_v1)
            cos_b = torch.zeros(ids.shape[0], len(blocks)); l2_b = torch.zeros_like(cos_b)
            for i in range(len(blocks)):
                x = store[("in", i)][:, SKIP:].float(); y = store[("out", i)][:, SKIP:].float()
                cos_b[:, i] = (1 - F.cosine_similarity(x, y, dim=-1)).mean(-1).cpu()
                l2_b[:, i] = ((y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).mean(-1).cpu()
                if check_v1 and i == len(blocks) - 1:
                    y_v1 = out.hidden_states[i + 1][:, SKIP:].float()   # what v1 used as "block output"
                    v1_gap.append((1 - F.cosine_similarity(x, y_v1, dim=-1)).mean(-1).cpu())
            cos_all.append(cos_b); l2_all.append(l2_b)
    finally:
        for h in handles:
            h.remove()
    res = {"bi": torch.cat(cos_all).numpy(), "rl2": torch.cat(l2_all).numpy()}
    if check_v1:
        res["v1_last"] = torch.cat(v1_gap).numpy()
    return res

t0 = time.time()
chunk_inf = {}
for lang in LANGS:
    cache = OUTPUT_DIR / f"chunk_influence_{lang}.npz"
    if cache.exists():
        saved = np.load(cache)
        chunk_inf[lang] = {k: saved[k] for k in saved.files}
        assert chunk_inf[lang]["bi"].shape == (CALIB_POOL_CHUNKS, N_LAYERS), (lang, chunk_inf[lang]["bi"].shape)
        print(lang, "influence loaded from cache")
    else:
        chunk_inf[lang] = per_chunk_influence(calib_pool[lang], check_v1=True)
        np.savez_compressed(cache, **chunk_inf[lang])
        print(lang, "influence computed and cached")
print("influence measured in %.0fs" % (time.time() - t0))

diag = pd.DataFrame([{"language": l,
                      "last_layer_BI_true": chunk_inf[l]["bi"][:, -1].mean(),
                      "last_layer_BI_as_in_v1": chunk_inf[l]["v1_last"].mean()} for l in LANGS])
print("Diagnostic: v1 measured the last layer against the post-final-norm hidden state")
display(diag)
diag.to_csv(OUTPUT_DIR / "diagnostic_v1_last_layer.csv", index=False)
""")

code(r"""
def percentile_ranks(values):
    ranks = pd.Series(np.asarray(values)).rank(method="average", ascending=True).to_numpy() - 1
    return ranks / max(1, len(ranks) - 1)

def language_scores(lang, metric, idx_by_lang=None):
    values = chunk_inf[lang][metric]
    if idx_by_lang is not None:
        values = values[idx_by_lang[lang]]
    return values.mean(0)

def condition_scores(condition, metric, idx_by_lang=None):
    if condition in SINGLE_CONDITIONS:
        return language_scores(condition, metric, idx_by_lang)

    per_language = np.stack([language_scores(lang, metric, idx_by_lang) for lang in AGG_LANGS])
    if condition == "lb_mean":
        return per_language.mean(0)

    ranked = np.stack([percentile_ranks(values) for values in per_language])
    if condition == "lb_rank_mean":
        return ranked.mean(0)
    if condition == "lb_minimax_rank":
        return ranked.max(0)
    raise ValueError(f"Unknown calibration condition: {condition}")

def select_condition(condition, metric, k, idx_by_lang=None):
    primary = condition_scores(condition, metric, idx_by_lang)
    layer_ids = np.arange(len(primary))
    if condition == "lb_minimax_rank":
        per_language = np.stack([language_scores(lang, metric, idx_by_lang) for lang in AGG_LANGS])
        secondary = np.stack([percentile_ranks(values) for values in per_language]).mean(0)
        order = np.lexsort((layer_ids, secondary, primary))
    else:
        order = np.lexsort((layer_ids, primary))
    return sorted(int(i) for i in order[:k])

influence = {m: pd.DataFrame({c: condition_scores(c, m) for c in CONDITIONS}) for m in ["bi", "rl2"]}
for m, df in influence.items():
    df.index.name = "layer"
    df.to_csv(OUTPUT_DIR / f"block_influence_{m}_by_language.csv")
influence["bi"].to_csv(OUTPUT_DIR / "block_influence_by_language.csv")
display(influence["bi"].style.format("{:.4f}").background_gradient(axis=0, cmap="Blues"))
print("LB mean equals the equal-weight mean over aggregation languages:",
      np.allclose(influence["bi"]["lb_mean"], influence["bi"][AGG_LANGS].mean(1)))
""")

md("## 4.1 Predeclared late-layer shift analysis")
code(r"""
ANALYSIS_LAYERS = np.arange(max(1, N_LAYERS - 1))  # exclude the final output-adjacent block
EARLY_LAYERS, MID_LAYERS, LATE_LAYERS = [np.asarray(x, dtype=int) for x in np.array_split(ANALYSIS_LAYERS, 3)]
LAYER_SEGMENT = {int(i): segment for segment, ids in
                 [("early", EARLY_LAYERS), ("middle", MID_LAYERS), ("late", LATE_LAYERS)] for i in ids}

def bootstrap_log_bi_ratio(target_lang, rng):
    en = np.clip(chunk_inf["en"]["bi"], 1e-12, None)
    target = np.clip(chunk_inf[target_lang]["bi"], 1e-12, None)
    observed = np.log(target.mean(0)) - np.log(en.mean(0))
    boot = np.empty((N_BOOT, N_LAYERS))
    chunk = 250
    for start in range(0, N_BOOT, chunk):
        size = min(chunk, N_BOOT - start)
        en_idx = rng.integers(0, len(en), size=(size, len(en)))
        target_idx = rng.integers(0, len(target), size=(size, len(target)))
        en_mean = en[en_idx].mean(1)
        target_mean = target[target_idx].mean(1)
        boot[start:start + size] = np.log(target_mean) - np.log(en_mean)
    return observed, boot

layer_shift_rows, late_shift_rows = [], []
rng_shift = np.random.default_rng(SEED + 4000)
for target_lang in TARGET_LANGS:
    observed, boot = bootstrap_log_bi_ratio(target_lang, rng_shift)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    for layer in range(N_LAYERS):
        layer_shift_rows.append({"model": MODEL_KEY, "target_language": target_lang, "layer": layer,
                                 "segment": LAYER_SEGMENT.get(layer, "excluded_final"),
                                 "log_bi_ratio_vs_en": observed[layer], "ci_low": lo[layer], "ci_high": hi[layer]})

    late_boot = boot[:, LATE_LAYERS].mean(1)
    middle_boot = boot[:, MID_LAYERS].mean(1)
    contrast_boot = late_boot - middle_boot
    late_mean = observed[LATE_LAYERS].mean()
    middle_mean = observed[MID_LAYERS].mean()
    late_shift_rows.append({"model": MODEL_KEY, "target_language": target_lang,
                            "middle_mean_log_bi_ratio": middle_mean,
                            "late_mean_log_bi_ratio": late_mean,
                            "late_geometric_shift_pct": 100 * (math.exp(late_mean) - 1),
                            "late_minus_middle_logratio": late_mean - middle_mean,
                            "contrast_ci_low": np.percentile(contrast_boot, 2.5),
                            "contrast_ci_high": np.percentile(contrast_boot, 97.5),
                            "late_stronger_than_middle": np.percentile(contrast_boot, 2.5) > 0})

layer_shift_df = pd.DataFrame(layer_shift_rows)
late_shift_df = pd.DataFrame(late_shift_rows)
layer_shift_df.to_csv(OUTPUT_DIR / "layer_bi_shift_bootstrap.csv", index=False)
late_shift_df.to_csv(OUTPUT_DIR / "late_layer_shift_summary.csv", index=False)
display(late_shift_df.round(4))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, m, lab in zip(axes, ["bi", "rl2"], ["Block Influence (1 − cosine)", "Relative L2 change"]):
    for c in LANGS:
        ax.plot(influence[m].index, influence[m][c], marker="o", ms=4, color=COLORS[c], label=c.upper())
    ax.set_yscale("log"); ax.set_xlabel("Transformer layer"); ax.set_ylabel(lab + " (log)")
axes[0].legend(frameon=False)
fig.suptitle(f"{MODELS[MODEL_KEY]}: layer influence by calibration language", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig1_layer_influence.png"); plt.show()
""")

md(r"""
## 5. RQ1 — do rankings differ across languages?

Spearman/Kendall over all layers are dominated by the depth trend (first/last layers always high), so we also
report the **Jaccard overlap of the actual pruning sets** (the k lowest layers), which is what matters for pruning.
""")
code(r"""
rank_rows = []
for m in ["bi", "rl2"]:
    df = influence[m]
    for a, b in itertools.combinations(SINGLE_CONDITIONS, 2):
        rho, rp = spearmanr(df[a], df[b]); tau, tp = kendalltau(df[a], df[b])
        row = {"metric": m, "pair": f"{a}-{b}", "spearman_rho": rho, "spearman_p": rp,
               "kendall_tau": tau, "kendall_p": tp}
        for k in PRUNE_LEVELS:
            A, B = set(select_condition(a, m, k)), set(select_condition(b, m, k))
            row[f"jaccard_k{k}"] = len(A & B) / len(A | B)
        rank_rows.append(row)
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUTPUT_DIR / "rank_agreement.csv", index=False)
display(rank_df.round(4))

pruning_sets = {m: {c: {k: select_condition(c, m, k) for k in PRUNE_LEVELS} for c in CONDITIONS}
                for m in ["bi", "rl2"]}
print("BI pruning sets:", json.dumps(pruning_sets["bi"]))
print("RL2 pruning sets:", json.dumps(pruning_sets["rl2"]))
json.dump(pruning_sets, open(OUTPUT_DIR / "pruning_sets.json", "w"), indent=1)

ranks = influence["bi"][SINGLE_CONDITIONS].rank()
corr = ranks.corr(method="spearman")
fig, ax = plt.subplots(figsize=(4.6, 3.9))
im = ax.imshow(corr.values, cmap="Blues", vmin=min(0.5, corr.values.min()), vmax=1)
ax.set_xticks(range(len(SINGLE_CONDITIONS))); ax.set_yticks(range(len(SINGLE_CONDITIONS)))
ax.set_xticklabels([c.upper() for c in SINGLE_CONDITIONS]); ax.set_yticklabels([c.upper() for c in SINGLE_CONDITIONS])
for i in range(len(SINGLE_CONDITIONS)):
    for j in range(len(SINGLE_CONDITIONS)):
        ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                color="white" if corr.values[i, j] > 0.9 else "#0b0b0b", fontsize=9)
ax.grid(False); ax.set_title("Spearman ρ of BI layer rankings", loc="left", fontsize=10)
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig2_rank_correlation.png"); plt.show()
""")

md(r"""
## 6. Pruning engine with per-chunk NLL cache

Pruning swaps the block list for a subset and restores it afterwards. Weights are never modified, so this is
equivalent to reloading a fresh model for every configuration (verified by re-computing the baseline at the end).
Results are cached per removed-layer set, so repeated sets are evaluated once.
""")
code(r"""
@contextmanager
def pruned(remove):
    remove = set(remove)
    set_transformer_blocks(model, [b for i, b in enumerate(ORIGINAL_BLOCKS) if i not in remove], BLOCK_PATH)
    try:
        yield model
    finally:
        set_transformer_blocks(model, ORIGINAL_BLOCKS, BLOCK_PATH)

@torch.no_grad()
def chunk_nll(chunks):
    # Returns mean token NLL per chunk (SEQ_LEN-1 predicted tokens each).
    out = []
    for s in range(0, len(chunks), EVAL_BATCH):
        ids = chunks[s:s + EVAL_BATCH].to(DEVICE)
        logits = model(input_ids=ids, use_cache=False).logits
        for r in range(ids.shape[0]):
            out.append(F.cross_entropy(logits[r, :-1].float(), ids[r, 1:]).item())
        del logits
    return np.array(out)

cache_payload = {"model_id": MODELS[MODEL_KEY], "revision": model_info["model_revision"],
                 "seq_len": SEQ_LEN,
                 "eval_article_ids": {l: eval_article_ids[l].tolist() for l in LANGS}}
EVAL_FINGERPRINT = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()[:16]
NLL_CACHE = {}
EVAL_STATS = {"computed": 0, "loaded": 0, "compute_seconds": 0.0}

def nll_cache_path(key):
    layer_tag = "none" if not key else "layers_" + "-".join(map(str, key))
    return NLL_CACHE_DIR / f"{EVAL_FINGERPRINT}__{layer_tag}.npz"

def evaluate_set(remove):
    key = tuple(sorted(int(x) for x in remove))
    if key not in NLL_CACHE:
        cache = nll_cache_path(key)
        if cache.exists():
            saved = np.load(cache)
            assert set(saved.files) == set(LANGS), (cache, saved.files)
            NLL_CACHE[key] = {l: saved[l] for l in LANGS}
            assert all(len(NLL_CACHE[key][l]) == EVAL_CHUNKS for l in LANGS), cache
            EVAL_STATS["loaded"] += 1
        else:
            eval_start = time.time()
            with pruned(key):
                NLL_CACHE[key] = {l: chunk_nll(eval_chunks[l]) for l in LANGS}
            np.savez_compressed(cache, **NLL_CACHE[key])
            elapsed = time.time() - eval_start
            EVAL_STATS["computed"] += 1
            EVAL_STATS["compute_seconds"] += elapsed
            avg = EVAL_STATS["compute_seconds"] / EVAL_STATS["computed"]
            print(f"evaluated {key or 'baseline'} in {elapsed:.1f}s "
                  f"({EVAL_STATS['computed']} new, {EVAL_STATS['loaded']} cached; {avg:.1f}s/config average)")
    return NLL_CACHE[key]

ppl = lambda nll: float(np.exp(nll.mean()))

t0 = time.time()
base_nll = evaluate_set(())
baseline_ppl = {l: ppl(base_nll[l]) for l in LANGS}
print("Baseline PPL:", {l: round(v, 3) for l, v in baseline_ppl.items()}, "(%.0fs)" % (time.time() - t0))

# Bits per character: PPL is not comparable across languages with different tokenisation, BPC is.
def pred_chars(chunks):
    return np.array([len(tokenizer.decode(c[1:].tolist(), skip_special_tokens=True)) for c in chunks])
EVAL_CHARS = {l: pred_chars(eval_chunks[l]) for l in LANGS}
def bpc(nll, l):
    return float((nll * (SEQ_LEN - 1)).sum() / math.log(2) / EVAL_CHARS[l].sum())
baseline_bpc = {l: bpc(base_nll[l], l) for l in LANGS}
pd.DataFrame([{"language": l, "baseline_ppl": baseline_ppl[l], "baseline_bpc": baseline_bpc[l],
               "eval_chars": int(EVAL_CHARS[l].sum())} for l in LANGS]).to_csv(
    OUTPUT_DIR / "baseline_metrics.csv", index=False)
print("Baseline bits/char:", {l: round(v, 4) for l, v in baseline_bpc.items()})
assert all(np.isfinite(base_nll[l]).all() for l in LANGS), "non-finite NLL: numerical problem"
""")

md("## 7. RQ2 — main pruning sweep (reference rankings from the full calibration pool)")
code(r"""
def result_rows(remove, **meta):
    nll = evaluate_set(remove)
    rows = []
    for l in LANGS:
        p = ppl(nll[l])
        rows.append({**meta, "removed_layer_ids": str(sorted(remove)), "evaluation_language": l,
                     "pruned_fraction_pct": 100 * len(remove) / N_LAYERS,
                     "baseline_ppl": baseline_ppl[l], "pruned_ppl": p,
                     "relative_ppl_degradation_pct": 100 * (p - baseline_ppl[l]) / baseline_ppl[l]})
    return rows

t0 = time.time()
main_rows = []
if RUN_REFERENCE_SWEEP:
    for c in CONDITIONS:
        for k in PRUNE_LEVELS:
            main_rows += result_rows(pruning_sets["bi"][c][k], model=MODEL_KEY, metric="bi",
                                     calibration=c, pruned_layers=k, budget="full_pool_diagnostic")
    for k in PRUNE_LEVELS:
        main_rows += result_rows(list(range(N_LAYERS - 1 - k, N_LAYERS - 1)), model=MODEL_KEY, metric="none",
                                 calibration="deep_block", pruned_layers=k, budget="calibration_free")
main_df = pd.DataFrame(main_rows)
main_df.to_csv(OUTPUT_DIR / "reference_pruning_results.csv", index=False)
print("done in %.0fs, unique configs evaluated: %d" % (time.time() - t0, len(NLL_CACHE)))

if len(main_df):
    pivot = main_df.pivot_table(index=["calibration", "pruned_layers"], columns="evaluation_language",
                                values="relative_ppl_degradation_pct")[LANGS]
    display(pivot.round(2))
""")

md(r"""
### Random-pruning control (10 seeds per k)
""")
code(r"""
t0 = time.time()
rand_rows = []
for k in PRUNE_LEVELS:
    for s in range(N_RANDOM_SEEDS):
        rem = sorted(random.Random(1000 + 97 * k + s).sample(range(N_LAYERS), k))
        rand_rows += result_rows(rem, model=MODEL_KEY, metric="random", calibration="random", pruned_layers=k, seed=s)
rand_df = pd.DataFrame(rand_rows)
rand_df.to_csv(OUTPUT_DIR / "random_pruning_results.csv", index=False)
rand_summary = rand_df.groupby(["pruned_layers", "evaluation_language"]).relative_ppl_degradation_pct.agg(
    ["median", "mean", "std", "min", "max"]).round(2)
display(rand_summary)
print("%.0fs" % (time.time() - t0))
""")

md(r"""
### Primary uncertainty analysis

Primary contrasts are evaluated after the equal-budget seeded sweep. Confidence intervals jointly resample
calibration seeds and independent evaluation articles; paired sign-flip tests receive Holm correction.
""")
code(r"""
print("Primary statistical contrasts are computed after the equal-budget seeded sweep.")
""")

md("## 8. Calibration seeds: ranking stability (Jaccard) and seeded pruning (mean ± std)")
code(r"""
def draw_subset(cond, budget, seed):
    r = np.random.default_rng(seed)
    if cond in AGG_CONDITIONS:
        assert budget % len(AGG_LANGS) == 0, (budget, AGG_LANGS)
        per = budget // len(AGG_LANGS)
        return {l: r.choice(CALIB_POOL_CHUNKS, per, replace=False) for l in AGG_LANGS}
    return {cond: r.choice(CALIB_POOL_CHUNKS, budget, replace=False)}

t0 = time.time()
stab_rows, seeded_rows = [], []
seed_sets = {c: [] for c in CONDITIONS}
for c in CONDITIONS:
    for s in range(N_SEEDS):
        idx = draw_subset(c, PRIMARY_BUDGET, 10_000 + s)
        sets = {k: select_condition(c, "bi", k, idx) for k in PRUNE_LEVELS}
        seed_sets[c].append(sets)
        for k in PRUNE_LEVELS:
            seeded_rows += result_rows(sets[k], model=MODEL_KEY, metric="bi", calibration=c,
                                       pruned_layers=k, seed=s, budget=PRIMARY_BUDGET,
                                       late_layers_selected=sum(i in set(range(max(0, N_LAYERS - 1 - (N_LAYERS - 1) // 3),
                                                                              N_LAYERS - 1)) for i in sets[k]))
        pd.DataFrame(seeded_rows).to_csv(OUTPUT_DIR / "primary_pruning_results.partial.csv", index=False)
        done = CONDITIONS.index(c) * N_SEEDS + s + 1
        total = len(CONDITIONS) * N_SEEDS
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"primary selector progress {done}/{total}; elapsed {elapsed / 60:.1f} min; ETA {eta / 60:.1f} min")
    for k in PRUNE_LEVELS:
        js = [len(set(a[k]) & set(b[k])) / len(set(a[k]) | set(b[k]))
              for a, b in itertools.combinations(seed_sets[c], 2)]
        ref = set(pruning_sets["bi"][c][k])
        jr = [len(set(a[k]) & ref) / len(set(a[k]) | ref) for a in seed_sets[c]]
        stab_rows.append({"calibration": c, "k": k, "budget_chunks": PRIMARY_BUDGET,
                          "pairwise_jaccard_mean": np.mean(js), "pairwise_jaccard_std": np.std(js),
                          "jaccard_vs_full_pool_mean": np.mean(jr), "n_seeds": N_SEEDS})
stability_df = pd.DataFrame(stab_rows)
stability_df.to_csv(OUTPUT_DIR / "ranking_stability.csv", index=False)
seeded_df = pd.DataFrame(seeded_rows)
seeded_df.to_csv(OUTPUT_DIR / "primary_pruning_results.csv", index=False)
json.dump({c: [{str(k): v for k, v in sets.items()} for sets in by_seed]
           for c, by_seed in seed_sets.items()}, open(OUTPUT_DIR / "primary_pruning_sets.json", "w"), indent=1)
display(stability_df.round(3))
seeded_summary = seeded_df.groupby(["calibration", "pruned_layers", "evaluation_language"]
    ).relative_ppl_degradation_pct.agg(["mean", "std"]).unstack("evaluation_language").round(2)
display(seeded_summary)

wide = seeded_df.pivot(index=["calibration", "pruned_layers", "seed"],
                       columns="evaluation_language", values="relative_ppl_degradation_pct").reset_index()
wide["target_mean_degradation_pct"] = wide[TARGET_LANGS].mean(axis=1)
wide["target_worst_degradation_pct"] = wide[TARGET_LANGS].max(axis=1)
strategy_summary = wide.groupby(["calibration", "pruned_layers"])[
    ["target_mean_degradation_pct", "target_worst_degradation_pct"]].agg(["mean", "std"]).reset_index()
strategy_summary.columns = ["calibration", "pruned_layers", "target_mean_mean", "target_mean_std",
                            "target_worst_mean", "target_worst_std"]
strategy_summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
print("%.0fs, cache size %d" % (time.time() - t0, len(NLL_CACHE)))
""")

md("## 9. Primary hierarchical bootstrap contrasts")
code(r"""
def seed_nll(condition, k, eval_language):
    return np.stack([evaluate_set(seed_sets[condition][s][k])[eval_language] for s in range(N_SEEDS)])

def hierarchical_bootstrap_delta(nll_a, nll_b, rng):
    d = np.asarray(nll_a) - np.asarray(nll_b)
    point = float(d.mean())
    boot = np.empty(N_BOOT)
    chunk = 250
    for start in range(0, N_BOOT, chunk):
        size = min(chunk, N_BOOT - start)
        seed_idx = rng.integers(0, d.shape[0], size=(size, d.shape[0]))
        article_idx = rng.integers(0, d.shape[1], size=(size, d.shape[1]))
        sampled_seeds = d[seed_idx]
        sampled = np.take_along_axis(sampled_seeds, article_idx[:, None, :], axis=2)
        boot[start:start + size] = sampled.mean(axis=(1, 2))

    article_delta = d.mean(0)
    null = np.empty(N_BOOT)
    for start in range(0, N_BOOT, chunk):
        size = min(chunk, N_BOOT - start)
        signs = rng.choice([-1.0, 1.0], size=(size, d.shape[1]))
        null[start:start + size] = (signs * article_delta).mean(1)
    p_value = (1 + np.count_nonzero(np.abs(null) >= abs(article_delta.mean()))) / (N_BOOT + 1)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float(p_value)

def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted

contrast_specs = []
for eval_lang in TARGET_LANGS:
    contrast_specs.append(("en_vs_matched", eval_lang, "en", eval_lang))
    for agg in AGG_CONDITIONS:
        contrast_specs.append(("en_vs_lb", eval_lang, "en", agg))
        contrast_specs.append(("lb_vs_matched", eval_lang, agg, eval_lang))
for eval_lang in LANGS:
    contrast_specs += [
        ("aggregator_ablation", eval_lang, "lb_mean", "lb_rank_mean"),
        ("aggregator_ablation", eval_lang, "lb_mean", "lb_minimax_rank"),
        ("aggregator_ablation", eval_lang, "lb_rank_mean", "lb_minimax_rank"),
    ]

rng = np.random.default_rng(SEED + 7000)
contrast_rows = []
for k in PRUNE_LEVELS:
    for family, eval_lang, a, b in contrast_specs:
        na, nb = seed_nll(a, k, eval_lang), seed_nll(b, k, eval_lang)
        mean, lo, hi, p_value = hierarchical_bootstrap_delta(na, nb, rng)
        same = np.mean([seed_sets[a][s][k] == seed_sets[b][s][k] for s in range(N_SEEDS)])
        contrast_rows.append({"model": MODEL_KEY, "metric": "bi", "k": k,
                              "eval_language": eval_lang, "contrast_family": family,
                              "condition_a": a, "condition_b": b,
                              "same_layer_set_fraction": same, "delta_logppl": mean,
                              "ci_low": lo, "ci_high": hi, "ppl_ratio": math.exp(mean),
                              "p_value": p_value})
contrast_df = pd.DataFrame(contrast_rows)
contrast_df["p_holm"] = np.nan
for family, indices in contrast_df.groupby("contrast_family").groups.items():
    contrast_df.loc[indices, "p_holm"] = holm_adjust(contrast_df.loc[indices, "p_value"])
contrast_df["significant_holm_005"] = contrast_df.p_holm < 0.05
contrast_df.to_csv(OUTPUT_DIR / "primary_bootstrap_contrasts.csv", index=False)
display(contrast_df.round(4))
""")

md("## 9. RQ3 — calibration-budget sweep (k = %d)" % 4)
code(r"""
t0 = time.time()
budget_rows = []
if RUN_BUDGET_SWEEP:
    for B in BUDGETS:
        for c in CONDITIONS:
            for s in range(N_SEEDS):
                idx = draw_subset(c, B, 20_000 + 31 * B + s)
                rem = select_condition(c, "bi", BUDGET_K, idx)
                ref = set(pruning_sets["bi"][c][BUDGET_K])
                j = len(set(rem) & ref) / len(set(rem) | ref)
                for row in result_rows(rem, model=MODEL_KEY, metric="bi", calibration=c, pruned_layers=BUDGET_K,
                                       seed=s, budget=B):
                    row["jaccard_vs_full_pool"] = j
                    budget_rows.append(row)
budget_df = pd.DataFrame(budget_rows)
budget_df.to_csv(OUTPUT_DIR / "calibration_budget_sweep.csv", index=False)
if len(budget_df):
    budget_summary = budget_df.groupby(["budget", "calibration", "evaluation_language"]).agg(
        deg_mean=("relative_ppl_degradation_pct", "mean"), deg_std=("relative_ppl_degradation_pct", "std"),
        jaccard_mean=("jaccard_vs_full_pool", "mean")).round(3)
    display(budget_summary.unstack("evaluation_language"))
else:
    print("Budget sweep skipped; set RUN_BUDGET_SWEEP=1 to enable it.")
print("%.0fs, cache size %d" % (time.time() - t0, len(NLL_CACHE)))
""")

md("## 10. Figures")
code(r"""
# Fig 3: degradation vs k per evaluation language, with random band
fig, axes = plt.subplots(1, len(LANGS), figsize=(4.2 * len(LANGS), 4.2), sharey=False)
axes = np.atleast_1d(axes)
for ax, L in zip(axes, LANGS):
    r = rand_df[rand_df.evaluation_language == L].groupby("pruned_layers").relative_ppl_degradation_pct
    q25, q50, q75 = r.quantile(0.25), r.median(), r.quantile(0.75)
    ax.fill_between(PRUNE_LEVELS, q25, q75, color=COLORS["random"], alpha=0.2, lw=0, label="Random (IQR)")
    ax.plot(PRUNE_LEVELS, q50, color=COLORS["random"], ls="--", lw=1.5, label="Random (median)")
    for c in CONDITIONS:
        sub = seeded_df[(seeded_df.calibration == c) & (seeded_df.evaluation_language == L)].groupby(
            "pruned_layers").relative_ppl_degradation_pct
        ax.errorbar(PRUNE_LEVELS, sub.mean(), yerr=sub.std(), marker="o", ms=6, capsize=3,
                    color=COLORS[c], label=f"{c.upper()} calib")
    if len(main_df):
        d = main_df[(main_df.calibration == "deep_block") & (main_df.evaluation_language == L)]
        ax.plot(d.pruned_layers, d.relative_ppl_degradation_pct, marker="s", ms=6, color=COLORS["deep_block"],
                lw=1.5, label="Deep block")
    ax.set_yscale("symlog", linthresh=1); ax.set_xticks(PRUNE_LEVELS)
    ax.set_title(f"Evaluated on {L.upper()}", loc="left", fontsize=11)
    ax.set_xlabel("Layers removed"); ax.set_ylabel("Relative PPL increase (%) — log")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{MODELS[MODEL_KEY]}: BI pruning (mean ± std over {N_SEEDS} calibration seeds, "
             f"{PRIMARY_BUDGET} total chunks) vs controls", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig3_ppl_degradation_vs_k.png"); plt.show()

# Fig 4: calibration x evaluation matrix (equal-budget seeded mean), one panel per k
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, k in zip(axes, PRUNE_LEVELS):
    mat = seeded_df[seeded_df.pruned_layers == k].pivot_table(index="calibration", columns="evaluation_language",
        values="relative_ppl_degradation_pct", aggfunc="mean").loc[CONDITIONS, LANGS]
    lv = np.log10(mat.values.clip(0.01)); lo, hi_ = lv.min(), max(lv.max(), lv.min() + 1e-9)
    im = ax.imshow(lv, cmap="Blues", vmin=lo, vmax=hi_)
    ax.set_xticks(range(len(LANGS))); ax.set_xticklabels([f"eval {l.upper()}" for l in LANGS])
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels([c.upper() for c in CONDITIONS])
    for i in range(len(CONDITIONS)):
        for j in range(len(LANGS)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if (lv[i, j] - lo) / (hi_ - lo) > 0.55 else "#0b0b0b")
    ax.grid(False); ax.set_title(f"k = {k}", loc="left")
fig.suptitle("Relative PPL increase by calibration (rows) × evaluation language (cols)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig4_calibration_x_evaluation.png"); plt.show()

# Fig 5: calibration budget sweep — stability and Indic degradation
if RUN_BUDGET_SWEEP and len(budget_df):
    fig, axes = plt.subplots(1, 1 + len(TARGET_LANGS), figsize=(4.2 * (1 + len(TARGET_LANGS)), 4))
    bs = budget_df.groupby(["budget", "calibration", "seed"]).jaccard_vs_full_pool.first().reset_index()
    for c in CONDITIONS:
        g = bs[bs.calibration == c].groupby("budget").jaccard_vs_full_pool
        axes[0].errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", capsize=3,
                         color=COLORS[c], label=c.upper())
        for ax, L in zip(axes[1:], TARGET_LANGS):
            g2 = budget_df[(budget_df.calibration == c) & (budget_df.evaluation_language == L)].groupby(
                "budget").relative_ppl_degradation_pct
            ax.errorbar(g2.mean().index, g2.mean(), yerr=g2.std(), marker="o", capsize=3,
                        color=COLORS[c], label=c.upper())
    axes[0].set_ylabel(f"Jaccard vs full-pool top-{BUDGET_K}"); axes[0].set_ylim(0, 1.05)
    for ax, L in zip(axes[1:], TARGET_LANGS):
        ax.set_ylabel(f"PPL increase on {L.upper()} (%)"); ax.set_yscale("symlog", linthresh=1)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Calibration budget (total 256-token chunks)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Calibration-budget sweep (k={BUDGET_K}, {N_SEEDS} seeds)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig5_budget_sweep.png"); plt.show()
""")

md(r"""
## 11. Optional appendix task - Belebele reading comprehension (EN / HI / MR / TA)

Belebele is fully parallel: the same 300 questions are asked in all four languages. This diagnostic is disabled
for the primary run because the revised claim concerns block selection and held-out perplexity. Enable it only
with `RUN_BELEBELE=1`.
""")
code(r"""
if RUN_BELEBELE:
    def load_belebele(lang):
        cache = DATA_CACHE / f"belebele_{lang}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        ds = load_dataset("facebook/belebele", BELEBELE_CODES[lang], split="test")
        rows = [{k: r[k] for k in ["link", "question_number", "flores_passage", "question", "mc_answer1",
                                   "mc_answer2", "mc_answer3", "mc_answer4", "correct_answer_num"]} for r in ds]
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows

    bele_raw = {l: {(r["link"], r["question_number"]): r for r in load_belebele(l)} for l in LANGS}
    common = sorted(set.intersection(*[set(v) for v in bele_raw.values()]))
    qkeys = [common[i] for i in np.random.default_rng(SEED).permutation(len(common))[:N_BELEBELE]]

    def bele_prompt(r):
        return (f"P: {r['flores_passage']}\nQ: {r['question']}\nA: {r['mc_answer1']}\nB: {r['mc_answer2']}\n"
                f"C: {r['mc_answer3']}\nD: {r['mc_answer4']}\nAnswer:")

    LETTER_IDS = [tokenizer.encode(" " + L, add_special_tokens=False)[-1] for L in "ABCD"]
    assert len(set(LETTER_IDS)) == 4, LETTER_IDS
    bele_ids = {l: [torch.tensor(tokenizer(bele_prompt(bele_raw[l][q]))["input_ids"]) for q in qkeys] for l in LANGS}
    bele_gold = {l: np.array([int(bele_raw[l][q]["correct_answer_num"]) - 1 for q in qkeys]) for l in LANGS}
    print("Belebele questions per language:", len(qkeys),
          "| mean prompt tokens:", {l: int(np.mean([len(x) for x in bele_ids[l]])) for l in LANGS})

    @torch.no_grad()
    def bele_correct(lang):
        preds = []
        for ids in bele_ids[lang]:
            out = model(input_ids=ids[None].to(DEVICE), use_cache=False, logits_to_keep=1)
            preds.append(int(out.logits[0, -1, LETTER_IDS].argmax()))
        return np.array(preds) == bele_gold[lang]

    BELE_CACHE = {}
    def bele_set(remove):
        key = tuple(sorted(int(x) for x in remove))
        if key not in BELE_CACHE:
            with pruned(key):
                BELE_CACHE[key] = {l: bele_correct(l) for l in LANGS}
        return BELE_CACHE[key]

    t0 = time.time()
    bele_rows = []
    def add_bele(remove, **meta):
        res = bele_set(remove)
        for l in LANGS:
            bele_rows.append({**meta, "removed_layer_ids": str(sorted(remove)), "evaluation_language": l,
                              "accuracy": res[l].mean(), "n_questions": len(res[l])})
    add_bele((), model=MODEL_KEY, calibration="baseline", pruned_layers=0)
    print("baseline done %.0fs" % (time.time() - t0))
    for c in CONDITIONS:
        for k in PRUNE_LEVELS:
            add_bele(pruning_sets["bi"][c][k], model=MODEL_KEY, calibration=c, pruned_layers=k)
    for k in PRUNE_LEVELS:
        for s in range(BELEBELE_RANDOM_SEEDS):
            rem = sorted(random.Random(1000 + 97 * k + s).sample(range(N_LAYERS), k))   # same sets as PPL control
            add_bele(rem, model=MODEL_KEY, calibration="random", pruned_layers=k, seed=s)
    bele_df = pd.DataFrame(bele_rows)
    bele_df.to_csv(OUTPUT_DIR / "belebele_results.csv", index=False)
    print("Belebele done in %.0fs, unique configs: %d" % (time.time() - t0, len(BELE_CACHE)))
    display(bele_df[bele_df.calibration != "random"].pivot_table(
        index=["calibration", "pruned_layers"], columns="evaluation_language", values="accuracy")[LANGS].round(3))

    # Paired bootstrap on accuracy: EN-calibrated vs language-matched pruning, same questions
    brows = []
    for k in PRUNE_LEVELS:
        for L in TARGET_LANGS:
            for a, b in [("en", L), ("en", "lb_mean")]:
                ca = bele_set(pruning_sets["bi"][a][k])[L].astype(float)
                cb = bele_set(pruning_sets["bi"][b][k])[L].astype(float)
                d = ca - cb
                bs = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
                brows.append({"k": k, "eval_language": L, "contrast": f"{a} vs {b}",
                              "same_layer_set": pruning_sets["bi"][a][k] == pruning_sets["bi"][b][k],
                              "acc_diff": d.mean(), "ci_low": np.percentile(bs, 2.5),
                              "ci_high": np.percentile(bs, 97.5)})
    bele_contrast_df = pd.DataFrame(brows)
    bele_contrast_df["significant"] = (bele_contrast_df.ci_low > 0) | (bele_contrast_df.ci_high < 0)
    bele_contrast_df.to_csv(OUTPUT_DIR / "belebele_bootstrap_contrasts.csv", index=False)
    display(bele_contrast_df.round(4))

    fig, axes = plt.subplots(1, len(LANGS), figsize=(4.2 * len(LANGS), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, LANGS):
        base_acc = bele_df[(bele_df.calibration == "baseline") & (bele_df.evaluation_language == L)].accuracy.iloc[0]
        ax.axhline(base_acc, color="#0b0b0b", lw=1, label="Unpruned")
        ax.axhline(0.25, color=COLORS["random"], lw=1, ls=":", label="Chance")
        r = bele_df[(bele_df.calibration == "random") & (bele_df.evaluation_language == L)].groupby("pruned_layers").accuracy
        ax.plot(PRUNE_LEVELS, r.mean(), color=COLORS["random"], ls="--", marker="x", label="Random pruning")
        for c in CONDITIONS:
            sub = bele_df[(bele_df.calibration == c) & (bele_df.evaluation_language == L)].sort_values("pruned_layers")
            ax.plot(sub.pruned_layers, sub.accuracy, marker="o", color=COLORS[c], label=f"{c.upper()} calib")
        ax.set_xticks(PRUNE_LEVELS); ax.set_xlabel("Layers removed"); ax.set_ylabel("Belebele accuracy")
        ax.set_title(f"Evaluated on {L.upper()}", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{MODELS[MODEL_KEY]}: Belebele accuracy after BI pruning (n={len(qkeys)} per language)",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig6_belebele.png"); plt.show()
else:
    print("Belebele skipped (RUN_BELEBELE=0)")
""")

md("## 12. Sanity check and experiment summary")
code(r"""
recheck = {l: ppl(chunk_nll(eval_chunks[l])) for l in LANGS}
assert all(abs(recheck[l] - baseline_ppl[l]) / baseline_ppl[l] < 1e-3 for l in LANGS), (recheck, baseline_ppl)
print("Model restored correctly after all pruning runs:", {l: round(v, 3) for l, v in recheck.items()})

summary = {**model_info, "transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "dtype": str(DTYPE),
           "seq_len": SEQ_LEN, "calib_pool_chunks": CALIB_POOL_CHUNKS, "eval_chunks": EVAL_CHUNKS,
           "languages": LANGS, "aggregation_languages": AGG_LANGS, "conditions": CONDITIONS,
           "n_seeds": N_SEEDS, "primary_budget": PRIMARY_BUDGET, "budgets": BUDGETS,
           "n_random_seeds": N_RANDOM_SEEDS, "seed": SEED, "baseline_ppl": baseline_ppl,
           "baseline_bpc": baseline_bpc, "unique_configs_evaluated": len(NLL_CACHE),
           "evaluation_cache_stats": EVAL_STATS,
           "eval_fingerprint": EVAL_FINGERPRINT, "run_reference_sweep": RUN_REFERENCE_SWEEP,
           "run_budget_sweep": RUN_BUDGET_SWEEP,
           "belebele": RUN_BELEBELE, "n_belebele": N_BELEBELE if RUN_BELEBELE else 0}
np.savez_compressed(OUTPUT_DIR / "per_chunk_nll.npz",
                    **{("-".join(map(str, k)) or "none") + "__" + l: v[l] for k, v in NLL_CACHE.items() for l in LANGS})
json.dump(summary, open(OUTPUT_DIR / "experiment_config.json", "w"), indent=2)
print("Saved to", OUTPUT_DIR)
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "accelerator": "GPU",
               "colab": {"gpuType": "A100", "provenance": []}}
from pathlib import Path as _P
nbf.write(nb, str(_P(__file__).resolve().parents[1] / "v4_robust_language_balanced_block_pruning.ipynb"))
print("written", len(cells), "cells")
