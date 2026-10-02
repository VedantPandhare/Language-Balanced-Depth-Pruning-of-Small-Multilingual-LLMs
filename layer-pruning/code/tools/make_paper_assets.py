"""Builds every figure (PDF) and table (.tex) used in the paper from the CSVs in ../results.

Usage (from the project root):  python code/tools/make_paper_assets.py
Outputs: paper/figures/*.pdf, paper/tables/*.tex, paper/tables/key_numbers.json
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen", "llama", "gemma"]
NAME = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-1B", "gemma": "Gemma-2-2B"}
LANGS = ["en", "hi", "mr"]
CONDS = ["en", "hi", "mr", "mixed"]
KS = [2, 4, 6]
COL = {"en": "#2a78d6", "hi": "#eb6834", "mr": "#1baf7a", "mixed": "#eda100",
       "random": "#8a8984", "deep_block": "#4a3aa7"}
LBL = {"en": "EN", "hi": "HI", "mr": "MR", "mixed": "Mixed"}

# Architecture facts from each model's config.json (decoder-block parameters computed exactly)
ARCH = {
    "qwen": dict(L=28, d=1536, ffn=8960, heads=12, kv=2, hd=128, vocab=151936,
                 block=1536 * 1536 + 1536 + 2 * (1536 * 256 + 256) + 1536 * 1536 + 3 * 1536 * 8960 + 2 * 1536),
    "llama": dict(L=16, d=2048, ffn=8192, heads=32, kv=8, hd=64, vocab=128256,
                  block=2 * 2048 * 2048 + 2 * 2048 * 512 + 3 * 2048 * 8192 + 2 * 2048),
    "gemma": dict(L=26, d=2304, ffn=9216, heads=8, kv=4, hd=256, vocab=256000,
                  block=2304 * 2048 + 2 * 2304 * 1024 + 2048 * 2304 + 3 * 2304 * 9216 + 4 * 2304),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def rd(m, f):
    return pd.read_csv(RES / m / f)


def rj(m, f):
    return json.loads((RES / m / f).read_text())


key = {}  # numbers quoted in the text

# ---------------------------------------------------------------- Table: models + tokenizer + baseline
rows = []
for m in MODELS:
    cfg = rj(m, "experiment_config.json")
    a = ARCH[m]
    tok = rd(m, "tokenizer_tokens_per_word.csv").set_index("language")
    base = rd(m, "baseline_metrics.csv").set_index("language")
    bele = rd(m, "belebele_results.csv")
    bele0 = bele[bele.calibration == "baseline"].set_index("evaluation_language").accuracy
    pct_block = a["block"] / cfg["parameters"] * 100
    rows.append(dict(m=m, params=cfg["parameters"], a=a, pct_block=pct_block, tok=tok, base=base, bele0=bele0))
    key[m] = {"params_B": cfg["parameters"] / 1e9, "block_params_M": a["block"] / 1e6,
              "pct_params_per_layer": pct_block,
              "pct_params_removed": {k: k * pct_block for k in KS},
              "pct_layers_removed": {k: 100 * k / a["L"] for k in KS},
              "fertility": tok.tokens_per_word_mean.to_dict(),
              "baseline_ppl": base.baseline_ppl.to_dict(), "baseline_bpc": base.baseline_bpc.to_dict(),
              "belebele_base": bele0.to_dict(), "dtype": cfg["dtype"], "gpu": cfg["gpu"],
              "transformers": cfg["transformers"], "torch": cfg["torch"]}

with open(TAB / "tab_models.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Evaluated models. $L$: decoder blocks; $\Delta P$: share of all parameters removed per pruned block; fertility: tokenizer tokens per whitespace word; BPC: baseline bits per character on held-out Wikipedia.}
\label{tab:models}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lccc ccc ccc}
\toprule
 & & & & \multicolumn{3}{c}{Fertility} & \multicolumn{3}{c}{Baseline BPC}\\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Model & Params & $L$ & $\Delta P$ & EN & HI & MR & EN & HI & MR\\
\midrule
""")
    for r in rows:
        t, b = r["tok"].tokens_per_word_mean, r["base"].baseline_bpc
        f.write(f"{NAME[r['m']]} & {r['params']/1e9:.2f}B & {r['a']['L']} & {r['pct_block']:.1f}\\% & "
                f"{t['en']:.2f} & {t['hi']:.2f} & {t['mr']:.2f} & {b['en']:.3f} & {b['hi']:.3f} & {b['mr']:.3f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table: rank agreement
with open(TAB / "tab_rank.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Agreement between BI layer rankings computed on different calibration languages: Spearman $\rho$, Kendall $\tau$ over all layers, and Jaccard overlap $J_k$ of the $k$ lowest-BI (pruned) layer sets.}
\label{tab:rank}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{llccccc}
\toprule
Model & Pair & $\rho$ & $\tau$ & $J_2$ & $J_4$ & $J_6$\\
\midrule
""")
    for i, m in enumerate(MODELS):
        r = rd(m, "rank_agreement.csv")
        r = r[(r.metric == "bi") & r.pair.isin(["en-hi", "en-mr", "hi-mr"])].set_index("pair")
        key[m]["rank"] = r[["spearman_rho", "kendall_tau", "jaccard_k2", "jaccard_k4", "jaccard_k6"]].to_dict("index")
        for j, p in enumerate(["en-hi", "en-mr", "hi-mr"]):
            x = r.loc[p]
            lead = f"\\multirow{{3}}{{*}}{{{NAME[m]}}}" if j == 0 else ""
            f.write(f"{lead} & {p.upper().replace('-', '--')} & {x.spearman_rho:.3f} & {x.kendall_tau:.3f} & "
                    f"{x.jaccard_k2:.2f} & {x.jaccard_k4:.2f} & {x.jaccard_k6:.2f}\\\\\n")
        if i < 2:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table: transfer gap (main result)
gap_rows = []
for m in MODELS:
    c = rd(m, "bootstrap_contrasts.csv")
    sets = rj(m, "pruning_sets.json")["bi"]
    for k in KS:
        for L in ["hi", "mr"]:
            x = c[(c.metric == "bi") & (c.k == k) & (c.eval_language == L) & (c.contrast == f"en vs {L}")].iloc[0]
            mx = c[(c.metric == "bi") & (c.k == k) & (c.eval_language == L) & (c.contrast == "en vs mixed")].iloc[0]
            gap_rows.append(dict(model=m, k=k, lang=L, ratio=math.exp(x.delta_logppl), lo=math.exp(x.ci_low),
                                 hi=math.exp(x.ci_high), sig=bool(x.significant), same=bool(x.same_layer_set),
                                 ratio_mixed=math.exp(mx.delta_logppl), sig_mixed=bool(mx.significant),
                                 set_en=sets["en"][str(k)], set_l=sets[L][str(k)]))
gap = pd.DataFrame(gap_rows)
gap.to_csv(TAB / "transfer_gap.csv", index=False)
key["transfer_gap"] = gap.to_dict("records")


def fmt_ratio(r):
    if r.same:
        return r"\multicolumn{1}{c}{=}"
    s = f"{r.ratio:.2f} [{r.lo:.2f},{r.hi:.2f}]"
    return f"\\textbf{{{s}}}" if r.sig and r.lo > 1 else s


with open(TAB / "tab_gap.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Calibration-language transfer gap $\mathrm{TG}_\ell(k)=\mathrm{PPL}_\ell(\mathcal{S}^{\mathrm{EN}}_k)/\mathrm{PPL}_\ell(\mathcal{S}^{\ell}_k)$ with 95\% paired-bootstrap CIs. Bold: CI entirely above 1 (English calibration significantly worse). ``='': both calibrations select the same layers.}
\label{tab:gap}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{llcc}
\toprule
Model & $k$ & $\mathrm{TG}_{\mathrm{HI}}$ [95\% CI] & $\mathrm{TG}_{\mathrm{MR}}$ [95\% CI]\\
\midrule
""")
    for i, m in enumerate(MODELS):
        for j, k in enumerate(KS):
            h = gap[(gap.model == m) & (gap.k == k) & (gap.lang == "hi")].iloc[0]
            r = gap[(gap.model == m) & (gap.k == k) & (gap.lang == "mr")].iloc[0]
            lead = f"\\multirow{{3}}{{*}}{{{NAME[m]}}}" if j == 0 else ""
            f.write(f"{lead} & {k} & {fmt_ratio(h)} & {fmt_ratio(r)}\\\\\n")
        if i < 2:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table: full degradation + controls
deg_rows = []
for m in MODELS:
    d = rd(m, "pruning_perplexity_results.csv")
    rnd = rd(m, "random_pruning_results.csv")
    for k in KS:
        for L in LANGS:
            row = dict(model=m, k=k, lang=L)
            for c in CONDS:
                row[c] = d[(d.metric == "bi") & (d.calibration == c) & (d.pruned_layers == k) &
                           (d.evaluation_language == L)].relative_ppl_degradation_pct.iloc[0]
            row["deep_block"] = d[(d.calibration == "deep_block") & (d.pruned_layers == k) &
                                  (d.evaluation_language == L)].relative_ppl_degradation_pct.iloc[0]
            rr = rnd[(rnd.pruned_layers == k) & (rnd.evaluation_language == L)].relative_ppl_degradation_pct
            row["random_median"] = rr.median()
            row["random_min"] = rr.min()
            deg_rows.append(row)
deg = pd.DataFrame(deg_rows)
deg.to_csv(TAB / "degradation_all.csv", index=False)


def pct(v):
    if v >= 1e5:
        e = int(math.floor(math.log10(v)))
        return f"{v/10**e:.1f}e{e}"
    if v >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:.1f}"


with open(TAB / "tab_degradation.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table*}[t]
\centering
\caption{Relative perplexity increase (\%) after removing $k$ blocks selected by BI under each calibration language, versus two calibration-free controls (median of 10 random layer sets; contiguous deep block preceding the final layer). Best BI calibration per column in bold; underlined: worst.}
\label{tab:deg}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{ll ccc ccc ccc}
\toprule
 & & \multicolumn{3}{c}{Evaluated on EN} & \multicolumn{3}{c}{Evaluated on HI} & \multicolumn{3}{c}{Evaluated on MR}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Model & Selection & $k{=}2$ & $k{=}4$ & $k{=}6$ & $k{=}2$ & $k{=}4$ & $k{=}6$ & $k{=}2$ & $k{=}4$ & $k{=}6$\\
\midrule
""")
    for i, m in enumerate(MODELS):
        sub = deg[deg.model == m]
        lines = [("en", "BI, EN calib."), ("hi", "BI, HI calib."), ("mr", "BI, MR calib."),
                 ("mixed", "BI, Mixed calib."), ("random_median", "Random (median)"), ("deep_block", "Deep block")]
        for j, (c, lab) in enumerate(lines):
            cells = []
            for L in LANGS:
                for k in KS:
                    r = sub[(sub.k == k) & (sub.lang == L)].iloc[0]
                    v = r[c]
                    s = pct(v)
                    if c in CONDS:
                        vals = [r[x] for x in CONDS]
                        if np.isclose(v, min(vals)):
                            s = f"\\textbf{{{s}}}"
                        elif np.isclose(v, max(vals)) and not np.isclose(max(vals), min(vals)):
                            s = f"\\underline{{{s}}}"
                    cells.append(s)
            lead = f"\\multirow{{6}}{{*}}{{{NAME[m]}}}" if j == 0 else ""
            f.write(f"{lead} & {lab} & " + " & ".join(cells) + "\\\\\n")
            if j == 3:
                f.write("\\cmidrule(lr){2-11}\n")
        if i < 2:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------------------------------------------------------- Regret analysis (robust calibration)
reg_rows = []
for _, r in deg.iterrows():
    logp = {c: math.log1p(r[c] / 100) for c in CONDS}       # log(PPL_pruned / PPL_base)
    best = min(logp.values())
    for c in CONDS:
        reg_rows.append(dict(model=r.model, k=r.k, lang=r.lang, cal=c, regret_pct=100 * (math.exp(logp[c] - best) - 1),
                             matched=(c == r.lang)))
reg = pd.DataFrame(reg_rows)
reg.to_csv(TAB / "regret_all.csv", index=False)
summ = reg.groupby("cal").regret_pct.agg(["mean", "median", "max"])
wins = reg[reg.regret_pct < 1e-9].groupby("cal").size().reindex(CONDS).fillna(0).astype(int)
# Indic-only view (deployment target)
ind = reg[reg.lang != "en"].groupby("cal").regret_pct.agg(["mean", "max"])
key["regret"] = {c: {"mean": summ.loc[c, "mean"], "max": summ.loc[c, "max"], "wins": int(wins[c]),
                     "indic_mean": ind.loc[c, "mean"], "indic_max": ind.loc[c, "max"]} for c in CONDS}
# matched calibration as a strategy (each eval language uses its own calibration)
mt = reg[reg.matched]
key["regret"]["matched"] = {"mean": mt.regret_pct.mean(), "max": mt.regret_pct.max()}
with open(TAB / "tab_regret.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Calibration regret: extra perplexity (\%) relative to the best of the four calibrations for the same model, $k$ and evaluation language, aggregated over all 27 cells (3 models $\times$ 3 $k$ $\times$ 3 languages) and over the 18 Indic cells. Wins: cells where the calibration is (jointly) best.}
\label{tab:regret}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{3}{c}{All 27 cells} & \multicolumn{2}{c}{Indic (18 cells)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Calibration & Mean & Max & Wins & Mean & Max\\
\midrule
""")
    for c in CONDS:
        x = key["regret"][c]
        f.write(f"{LBL[c]} & {x['mean']:.1f} & {x['max']:.1f} & {x['wins']}/27 & {x['indic_mean']:.1f} & {x['indic_max']:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table: Belebele
with open(TAB / "tab_belebele.tex", "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Zero-shot Belebele accuracy (\%, 300 parallel questions per language, chance $=25$\%) of the unpruned model and after removing $k{=}2$ blocks (BI with EN or Mixed calibration; mean of two random sets).}
\label{tab:belebele}
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{ll ccc}
\toprule
Model & Variant & EN & HI & MR\\
\midrule
""")
    for i, m in enumerate(MODELS):
        b = rd(m, "belebele_results.csv")
        def acc(cal, k):
            s = b[(b.calibration == cal) & (b.pruned_layers == k)]
            return s.groupby("evaluation_language").accuracy.mean()
        variants = [("Unpruned", acc("baseline", 0)), ("BI, EN calib.", acc("en", 2)),
                    ("BI, Mixed calib.", acc("mixed", 2)), ("Random", acc("random", 2))]
        key[m]["belebele"] = {v: s.to_dict() for v, s in variants}
        for j, (v, s) in enumerate(variants):
            lead = f"\\multirow{{4}}{{*}}{{{NAME[m]}}}" if j == 0 else ""
            f.write(f"{lead} & {v} & " + " & ".join(f"{100*s[L]:.1f}" for L in LANGS) + "\\\\\n")
        if i < 2:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # max belebele contrast significance
for m in MODELS:
    bc = rd(m, "belebele_bootstrap_contrasts.csv")
    key[m]["belebele_any_significant"] = bool(bc.significant.any())

# ---------------------------------------------------------------- Stability
for m in MODELS:
    s = rd(m, "ranking_stability.csv")
    key[m]["stability"] = s.set_index(["calibration", "k"]).pairwise_jaccard_mean.unstack().to_dict()
    sp = rd(m, "seeded_pruning_results.csv")
    key[m]["seeded"] = sp.groupby(["calibration", "pruned_layers", "evaluation_language"]
                                  ).relative_ppl_degradation_pct.agg(["mean", "std"]).reset_index().to_dict("records")
    bs = rd(m, "calibration_budget_sweep.csv")
    key[m]["budget_jaccard"] = bs.groupby(["budget", "calibration"]).jaccard_vs_full_pool.mean().unstack().to_dict()
    key[m]["diag_last_layer"] = rd(m, "diagnostic_v1_last_layer.csv").to_dict("records")

# ---------------------------------------------------------------- Figure: layer influence profiles
fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95))
for ax, m in zip(axes, MODELS):
    inf = rd(m, "block_influence_bi_by_language.csv").set_index("layer")
    for L in LANGS:
        ax.plot(inf.index, inf[L], marker="o", color=COL[L], label=LBL[L])
    ax.set_yscale("log")
    ax.set_title(NAME[m], loc="left")
    ax.set_xlabel("Block index $l$")
    sets = rj(m, "pruning_sets.json")["bi"]
    only_en = sorted(set(sets["en"]["6"]) - set(sets["hi"]["6"]) | set(sets["en"]["2"]) - set(sets["hi"]["2"]))
    for l in only_en:
        ax.axvspan(l - 0.5, l + 0.5, color=COL["en"], alpha=0.10, lw=0)
axes[0].set_ylabel(r"BI$_l$ = 1 $-$ cos (log)")
axes[0].legend(frameon=False, loc="upper center", ncol=3)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / "fig_influence.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: relative BI shift vs depth
MCOL = {"qwen": "#4a3aa7", "llama": "#e34948", "gemma": "#52514e"}   # model identity, distinct from language hues
fig, ax = plt.subplots(figsize=(3.45, 1.9))
marker = {"qwen": "o", "llama": "s", "gemma": "^"}
for m in MODELS:
    inf = rd(m, "block_influence_bi_by_language.csv").set_index("layer")
    depth = inf.index / (len(inf) - 1)
    shift = 100 * ((inf["hi"] + inf["mr"]) / 2 - inf["en"]) / inf["en"]
    mid = (depth > 0.3) & (depth < 0.6); late = (depth >= 0.6) & (depth < 1.0)
    key[m]["shift_mid_mean"] = float(shift[mid].mean()); key[m]["shift_late_mean"] = float(shift[late].mean())
    key[m]["shift_late_max"] = float(shift[late].max()); key[m]["shift_late_argmax_layer"] = int(shift[late].idxmax())
    ax.plot(depth, shift, marker=marker[m], color=MCOL[m],
            label=NAME[m])
ax.axhline(0, color="#0b0b0b", lw=0.7)
ax.set_xlabel("Relative depth $l/(L-1)$")
ax.set_ylabel(r"Indic vs. EN BI shift (%)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig_shift.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: transfer gap bars
fig, ax = plt.subplots(figsize=(3.45, 1.95))
xt, xl = [], []
x = 0
for m in MODELS:
    for k in KS:
        for o, L in [(-0.19, "hi"), (0.19, "mr")]:
            r = gap[(gap.model == m) & (gap.k == k) & (gap.lang == L)].iloc[0]
            ax.bar(x + o, r.ratio, width=0.36, color=COL[L], label=LBL[L] if (m == "qwen" and k == 2) else None,
                   edgecolor="white", linewidth=0.4)
            if not r.same:
                ax.errorbar(x + o, r.ratio, yerr=[[r.ratio - r.lo], [r.hi - r.ratio]], color="#0b0b0b",
                            capsize=1.5, lw=0.7)
        xt.append(x)
        xl.append(f"{k}")
        x += 1
    x += 0.5
ax.axhline(1, color="#0b0b0b", lw=0.7)
ax.set_xticks(xt)
ax.set_xticklabels(xl)
ax.set_ylabel(r"TG$_\ell(k)$ (PPL ratio)")
for i, m in enumerate(MODELS):
    ax.text(i * 3.5 + 1, -0.3, NAME[m], ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7)
ax.set_xlabel("")
ax.legend(frameon=False, loc="upper left", title="Evaluated on", title_fontsize=7)
ax.set_ylim(0.8, max(2.7, gap.hi.max() + 0.1))
fig.tight_layout()
fig.subplots_adjust(bottom=0.27)
fig.savefig(FIG / "fig_gap.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: BI vs controls on Marathi
fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95))
for ax, m in zip(axes, MODELS):
    rnd = rd(m, "random_pruning_results.csv")
    rr = rnd[rnd.evaluation_language == "mr"].groupby("pruned_layers").relative_ppl_degradation_pct
    ax.fill_between(KS, rr.quantile(0.25), rr.quantile(0.75), color=COL["random"], alpha=0.25, lw=0,
                    label="Random (IQR)")
    ax.plot(KS, rr.median(), color=COL["random"], ls="--", label="Random (median)")
    sub = deg[(deg.model == m) & (deg.lang == "mr")].sort_values("k")
    ax.plot(KS, sub.deep_block, color=COL["deep_block"], marker="s", label="Deep block")
    for c in CONDS:
        ax.plot(KS, sub[c], color=COL[c], marker="o", label=f"BI, {LBL[c]} calib.",
                ls="-" if c in ("en", "mixed") else ":")
    ax.set_yscale("log")
    ax.set_xticks(KS)
    ax.set_xlabel("Blocks removed $k$")
    ax.set_title(NAME[m], loc="left")
axes[0].set_ylabel("PPL increase on MR (%, log)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / "fig_controls_mr.pdf")
plt.close(fig)

(TAB / "key_numbers.json").write_text(json.dumps(key, indent=1, default=float))
print("Assets written to", FIG, "and", TAB)
print(json.dumps({"regret": key["regret"]}, indent=1, default=float))
print(gap[["model", "k", "lang", "ratio", "lo", "hi", "sig", "same", "ratio_mixed", "set_en", "set_l"]].round(3).to_string())
for m in MODELS:
    print(m, {k: round(v, 2) for k, v in key[m]["pct_params_removed"].items()}, key[m]["stability"])
