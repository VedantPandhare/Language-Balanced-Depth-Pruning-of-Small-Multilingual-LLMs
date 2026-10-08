# Language-Aware Layer Pruning in Small Multilingual LLMs
## IEEE-oriented conceptual and research guide

**Working title:**  
**Do Small Multilingual LLMs Share Redundant Layers Across Languages? Language-Aware Layer Pruning for English, Hindi, and Marathi**

---

## 1. The problem

Large language models are expensive because every input normally passes through many Transformer layers.

A natural question is:

> Are all layers equally necessary for every language?

Suppose a small multilingual LLM has 16 or 28 Transformer blocks. If some blocks change their hidden representation only slightly, those blocks may look like pruning candidates.

But there is a deeper issue.

A layer can appear unimportant when measured on English text and still be important for Hindi or Marathi.

Therefore, the real research problem is:

> **Does layer redundancy depend on the language used to measure it?**

This creates a specific research gap in multilingual structural pruning.

---

# 2. The research idea in one sentence

Measure layer influence separately on English, Hindi, and Marathi, remove the layers ranked as least influential, and test whether English-based pruning preserves Indic-language performance as well as language-aware pruning.

---

# 3. Why this is a research problem

Most pruning pipelines need some calibration data.

Calibration data tells the pruning method which parameters, neurons, heads, or layers appear less important.

A common implicit assumption is:

> The calibration distribution is representative enough for the target use case.

For multilingual models, that assumption may fail.

English, Hindi, and Marathi differ in:
- vocabulary
- script
- morphology
- word segmentation
- tokenizer behavior
- token frequency
- sentence structure
- amount and quality of pretraining data

Therefore, a layer that has low average influence on English can have a different role for Indic text.

The project turns this into an experimentally testable question rather than assuming that multilingual models have language-independent redundancy.

---

# 4. Transformer architecture needed for this paper

A causal language model processes a sequence of tokens through a stack of Transformer blocks.

A simplified view is:

```text
Tokens
  |
  v
Embedding
  |
  v
Transformer Block 1
  |
  v
Transformer Block 2
  |
  v
...
  |
  v
Transformer Block N
  |
  v
Language-model head
  |
  v
Next-token probabilities
```

Each Transformer block normally contains:
1. normalization
2. self-attention
3. residual connection
4. normalization
5. feed-forward network
6. residual connection

Exact ordering differs between architectures.

The important point for this project is that the model has a sequence of blocks:

```text
H0 -> Block1 -> H1 -> Block2 -> H2 -> ... -> BlockN -> HN
```

The research asks whether some of these blocks can be removed with relatively small performance loss.

---

# 5. What is layer pruning?

Layer pruning means removing complete Transformer blocks.

For example:

```text
Original:

B1 B2 B3 B4 B5 B6 B7 B8

Remove B2 and B6:

B1 B3 B4 B5 B7 B8
```

The goal is to reduce model depth.

This is different from:
- quantization
- weight pruning
- neuron pruning
- attention-head pruning
- knowledge distillation

Layer pruning changes the number of Transformer blocks.

---

# 6. What does "redundant layer" mean here?

Be careful with terminology.

This project does **not** initially prove that a layer is mathematically redundant.

Instead, it defines an operational proxy:

> A layer is a pruning candidate if it produces a relatively small average change in hidden representations on calibration data.

That distinction is important for an IEEE paper.

Do not write:

> "Layer 5 is redundant."

Prefer:

> "Layer 5 exhibited low block influence under the specified calibration distribution and was therefore selected as a pruning candidate."

Then actual pruning and perplexity evaluation provide evidence about whether that candidate was useful.

---

# 7. Block Influence

For a Transformer block, let:

- `H_in` = hidden representation entering the block
- `H_out` = hidden representation leaving the block

For every token:

```text
cosine(H_in, H_out)
```

measures how directionally similar the representations are.

The proposed influence proxy is:

```text
Block Influence = 1 - cosine(H_in, H_out)
```

Interpretation:

```text
High cosine
    |
    v
Input and output are similar
    |
    v
Low influence

Low cosine
    |
    v
Input and output changed more
    |
    v
High influence
```

Therefore:

```text
Low Block Influence -> pruning candidate
High Block Influence -> stronger candidate to keep
```

This is only a proxy.

---

# 8. Why cosine similarity?

Hidden states are high-dimensional vectors.

Cosine similarity focuses on the angle between two vectors rather than their absolute magnitude.

For vectors `x` and `y`:

```text
cos(x,y) = (x · y) / (||x|| ||y||)
```

The proposed influence is:

```text
I = 1 - cos(x,y)
```

This gives a simple normalized signal that can be averaged over tokens.

A future extension can compare this with:

```text
Relative L2 change =
||H_out - H_in|| / ||H_in||
```

If both metrics produce similar conclusions, the result becomes more robust to the choice of influence metric.

---

# 9. The central research hypothesis

### H1

Layer-importance rankings are not identical across English, Hindi, and Marathi.

### H2

Pruning layers selected using English-only calibration can cause larger Indic-language degradation than pruning using language-matched calibration.

### H3

Balanced multilingual calibration can reduce the mismatch between English and Indic pruning decisions.

### H4

Some differences in pruning behavior are associated with tokenizer token-per-word statistics.

These are hypotheses. The experiment must be allowed to disprove them.

---

# 10. Research questions

## RQ1

**Do Transformer-layer importance rankings differ across English, Hindi, and Marathi?**

Measure:
- Block Influence
- Spearman rank correlation
- Kendall tau
- ranking stability

---

## RQ2

**How does calibration language affect the performance of layer pruning?**

Compare:

```text
English calibration
Hindi calibration
Marathi calibration
Balanced multilingual calibration
```

Evaluate every pruned model on:

```text
English
Hindi
Marathi
```

---

## RQ3

**Can a small balanced multilingual calibration set provide language-robust pruning decisions?**

Compare English-only and mixed calibration.

A useful follow-up is a calibration-budget sweep:

```text
1 sample/language
5 samples/language
10 samples/language
20 samples/language
50 samples/language
100 samples/language
```

This can turn the paper from a simple language-comparison study into a practical calibration-efficiency study.

---

## RQ4

**Are cross-language pruning differences associated with tokenizer behavior?**

Measure:

```text
tokens per whitespace-separated word
```

for each language.

Treat this as a correlation or explanatory analysis, not causal proof.

---

# 11. Experimental design

The primary experiment contains four calibration conditions:

| Calibration | Purpose |
|---|---|
| English | Tests the common English-calibration assumption |
| Hindi | Language-matched Indic calibration |
| Marathi | Language-matched Indic calibration |
| Mixed | Tests multilingual calibration |

Three pruning levels:

```text
2 layers
4 layers
6 layers
```

Three evaluation languages:

```text
English
Hindi
Marathi
```

Therefore:

```text
4 calibration conditions
x 3 pruning levels
x 3 evaluation languages
= 36 evaluation conditions per model
```

With three models:

```text
36 x 3 = 108 conditions
```

before adding random baselines and multiple seeds.

---

# 12. Recommended model strategy

Use one model for the main experiment and additional models for replication.

### Primary

**Qwen2.5-1.5B-Instruct**

Why:
- small enough for Colab experimentation
- modern decoder-only Transformer
- multilingual use case
- practical model size

### Replication

**Llama-3.2-1B**

Use it to test whether the observed phenomenon is specific to Qwen.

### Additional replication

**Gemma-2-2B**

Use it to test whether the result generalizes across another architecture family.

The final paper should report exact model identifiers and versions.

---

# 13. Dataset strategy

The notebook uses Wikimedia Wikipedia subsets for:

```text
English
Hindi
Marathi
```

The important experimental distinction is:

```text
Calibration set
        |
        v
Choose pruning candidates

Evaluation set
        |
        v
Measure perplexity
```

These must be separated.

If the same examples are used for calibration and evaluation, the evaluation can become less convincing.

---

# 14. Why perplexity?

A causal language model predicts the next token.

Perplexity is derived from average negative log-likelihood:

```text
PPL = exp(average token NLL)
```

Lower perplexity generally means the model predicts the evaluation text better.

For this project, the most useful measurement is not only absolute PPL but the change caused by pruning.

Define:

```text
Relative PPL degradation (%)
=
100 x (PPL_pruned - PPL_baseline) / PPL_baseline
```

Example:

```text
Baseline PPL = 20
Pruned PPL   = 22

Degradation = 100 x (22 - 20) / 20
            = 10%
```

---

# 15. The most important comparison

Suppose four layers are removed.

We could obtain:

```text
English calibration -> English evaluation
English calibration -> Hindi evaluation
English calibration -> Marathi evaluation

Hindi calibration -> Hindi evaluation
Marathi calibration -> Marathi evaluation

Mixed calibration -> English/Hindi/Marathi evaluation
```

The central question is whether:

```text
English calibration
        |
        v
English performance preserved
but
Indic performance degrades more
```

If that pattern appears consistently, it supports the motivation for language-aware calibration.

---

# 16. Rank agreement

Suppose English ranks layers:

```text
L3, L7, L5, L1, L8, ...
```

Hindi ranks them:

```text
L7, L3, L1, L8, L5, ...
```

The exact order differs.

Spearman correlation measures whether the rankings are monotonic relative to each other.

Kendall tau measures pairwise ranking agreement.

Use both because they capture related but different aspects of ranking consistency.

Do not interpret a single correlation as proof of functional similarity.

---

# 17. Ranking stability

A major concern is calibration noise.

Suppose 20 English samples say:

```text
Remove layers: 3, 7, 10, 12
```

A different random 20-sample calibration set says:

```text
Remove layers: 2, 7, 9, 12
```

The ranking may not be stable.

Use Jaccard similarity:

```text
J(A,B) = |A intersection B| / |A union B|
```

For a final paper, calculate this across multiple seeds.

Report:

```text
mean Jaccard ± standard deviation
```

---

# 18. Random pruning baseline

This is important.

Suppose influence-based pruning removes four layers.

Compare it with random removal of four layers.

If influence-based pruning produces no better preservation of performance than random pruning, the proposed ranking signal is not demonstrating useful selection.

A stronger experiment uses several random seeds.

For example:

```text
seed 1
seed 2
seed 3
seed 4
seed 5
```

Then report the mean and variation.

---

# 19. Why language matching matters

The proposed mechanism is:

```text
Language
   |
   v
Tokenization + linguistic distribution
   |
   v
Hidden-state behavior
   |
   v
Measured layer influence
   |
   v
Pruning ranking
   |
   v
Final performance
```

The project does not assume that every difference is caused by language itself.

Possible confounders include:
- tokenization
- dataset domain
- text quality
- sequence length
- sample size
- model pretraining mixture
- script
- morphology
- calibration noise

These should appear in the limitations section.

---

# 20. Tokenizer analysis

For each language calculate:

```text
tokens per word
=
number of tokenizer tokens
/
number of whitespace-separated words
```

A higher value means the tokenizer represents the text with more tokens per word under this simple measure.

This matters because the Transformer operates on tokens, not human-defined words.

However:

> Token-per-word differences do not establish that tokenization causes layer redundancy differences.

Treat it as an explanatory variable.

A stronger future analysis could control for:
- equal token budgets
- equal sequence counts
- equal document lengths
- matched domains

---

# 21. Token budget is more important than document count

Do not compare:

```text
100 English documents
vs
100 Hindi documents
```

and assume they provide equal calibration information.

Languages may produce different numbers of tokens.

For the final experiment, report:

```text
number of calibration tokens
number of evaluation tokens
sequence length
number of sequences
```

Prefer matching token budgets where practical.

This is especially important for an IEEE paper because otherwise the language comparison has a potential exposure imbalance.

---

# 22. The strongest experimental version

The final study should ideally contain:

### Models
- Qwen2.5-1.5B
- Llama-3.2-1B
- Gemma-2-2B

### Languages
- English
- Hindi
- Marathi

### Calibration
- English
- Hindi
- Marathi
- balanced mixed

### Pruning
- 2
- 4
- 6 layers

### Baselines
- influence-based pruning
- random pruning

### Repetitions
- multiple calibration seeds

### Metrics
- Block Influence
- Spearman
- Kendall tau
- perplexity
- relative PPL degradation
- Jaccard stability
- tokens per word

### Optional downstream evaluation
Use one common sentiment task covering Hindi and Marathi, such as an Indic sentiment benchmark.

This is useful because perplexity is a language-modeling metric, while downstream performance tests whether pruning affects an actual task.

---

# 23. A very important methodological issue

The simplest influence metric has a limitation.

A layer can produce a small change:

```text
H_in ≈ H_out
```

but still be important because the small transformation may be strategically useful to later layers.

Therefore:

```text
Low influence
≠
Guaranteed safe to remove
```

The actual pruning experiment is essential.

This distinction should be explicit in the paper.

---

# 24. Another important issue: sequential pruning

Do not do this:

```text
prune 2
then prune another 2
then prune another 2
```

and treat the results as independent 2/4/6 layer experiments.

Instead:

```text
Fresh model -> remove 2
Fresh model -> remove 4
Fresh model -> remove 6
```

Otherwise the 6-layer model has been affected by earlier pruning decisions.

The notebook follows the fresh-model principle.

---

# 25. Computational experiment flow

```text
Load model
   |
   v
Load separate calibration/evaluation data
   |
   v
Measure Block Influence
   |
   +------ English
   |
   +------ Hindi
   |
   +------ Marathi
   |
   +------ Mixed
   |
   v
Create layer rankings
   |
   v
Compare rankings
   |
   +------ Spearman
   +------ Kendall
   +------ Stability
   |
   v
Select 2/4/6 candidate layers
   |
   v
Reload fresh model
   |
   v
Prune selected layers
   |
   v
Evaluate PPL on EN/HI/MR
   |
   v
Compare against baseline
   |
   v
Compare against random pruning
```

---

# 26. Expected result patterns

Do not assume the result before running the experiment.

Possible outcomes include:

### Outcome A: strong language dependence

English, Hindi, and Marathi rankings differ substantially, and English-calibrated pruning damages Indic performance more.

This would support the central hypothesis.

### Outcome B: mostly language-independent redundancy

Rankings are highly similar and pruning behaves similarly across languages.

This would challenge the hypothesis and still be a useful negative result.

### Outcome C: weak ranking difference but strong performance difference

This could indicate that the influence proxy does not fully capture functional importance.

That would motivate better importance metrics.

### Outcome D: tokenizer-related effect

Differences may correlate with tokenization statistics.

This would motivate a controlled follow-up.

A good paper reports whichever outcome the data support.

---

# 27. Statistical reporting

Avoid reporting only one number.

For final experiments, report:

```text
mean
standard deviation
sample count
```

where repeated runs exist.

For example:

```text
English-calibrated pruning:
Hindi PPL degradation = mean ± std
```

For rank correlations, report the coefficient and clearly state what samples/layers produced it.

For multiple model/language comparisons, organize results into tables instead of discussing every number in prose.

---

# 28. Recommended main tables

## Table 1: Model configuration

| Model | Parameters | Layers | Hidden size | Attention configuration |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B | report exact | report | report | report |
| Llama-3.2-1B | report exact | report | report | report |
| Gemma-2-2B | report exact | report | report | report |

Use exact configuration values from the model configuration files.

---

## Table 2: Cross-language ranking agreement

| Pair | Spearman | Kendall tau |
|---|---:|---:|
| EN-HI | ... | ... |
| EN-MR | ... | ... |
| HI-MR | ... | ... |

---

## Table 3: Perplexity degradation

| Calibration | Pruned | Eval EN | Eval HI | Eval MR |
|---|---:|---:|---:|---:|
| EN | 2 | ... | ... | ... |
| EN | 4 | ... | ... | ... |
| EN | 6 | ... | ... | ... |
| HI | 2 | ... | ... | ... |
| ... | ... | ... | ... | ... |

---

## Table 4: Stability

| Language | k | Jaccard mean | Std |
|---|---:|---:|---:|
| EN | 2 | ... | ... |
| EN | 4 | ... | ... |
| EN | 6 | ... | ... |
| HI | ... | ... | ... |

---

# 29. Recommended main figures

### Figure 1

Layer influence by language.

### Figure 2

Rank correlation heatmap.

### Figure 3

Relative PPL degradation versus number of pruned layers.

### Figure 4

Calibration condition versus evaluation language.

### Figure 5

Ranking stability across seeds.

### Figure 6

Optional tokenizer tokens-per-word comparison.

---

# 30. IEEE paper structure

A practical IEEE-style structure is:

```text
Abstract
Keywords

I. Introduction
II. Related Work
III. Research Gap and Research Questions
IV. Methodology
V. Experimental Setup
VI. Results
VII. Discussion
VIII. Threats to Validity and Limitations
IX. Conclusion and Future Work
References
```

---

# 31. Abstract template

Do not fill in numerical claims until the experiment is complete.

> **Abstract—** Transformer layer pruning offers a potential approach for reducing the computational cost of autoregressive language models. However, pruning decisions for multilingual models may depend on the calibration language used to estimate layer importance. This study investigates whether layer redundancy in small multilingual language models is language-dependent, focusing on English, Hindi, and Marathi. We estimate layer influence using the change in hidden representations across Transformer blocks and compare layer-importance rankings across languages. We then evaluate structured layer pruning under English-only, language-matched, and balanced multilingual calibration conditions using held-out perplexity and downstream evaluation. The study additionally examines tokenizer token-per-word statistics and ranking stability across calibration samples. The experiments are designed to determine whether English-calibrated pruning transfers reliably to Indic languages and whether a small multilingual calibration set can improve pruning robustness. [Insert final quantitative findings here.] The results provide evidence regarding the extent to which calibration language should be considered when applying structural pruning to multilingual language models.

---

# 32. Introduction logic

The introduction should move in this order:

### Paragraph 1
LLMs are computationally expensive.

### Paragraph 2
Layer pruning can reduce model depth and inference cost.

### Paragraph 3
Pruning needs an importance signal and calibration data.

### Paragraph 4
Multilingual models create a potential problem: importance may depend on language.

### Paragraph 5
Existing pruning evaluations may not explicitly test whether the same layers are redundant across languages.

### Paragraph 6
This work studies English, Hindi, and Marathi in small multilingual LLMs.

### Paragraph 7
State contributions.

Possible contribution wording:

> The contributions of this work are:
>
> 1. A cross-language analysis of Transformer-layer influence in small multilingual language models.
> 2. An evaluation of English-only, language-matched, and balanced multilingual calibration for structured layer pruning.
> 3. A quantitative comparison of layer-ranking agreement across English, Hindi, and Marathi.
> 4. An analysis of pruning stability and tokenizer token-per-word behavior.
> 5. An empirical assessment of whether multilingual calibration can provide more robust pruning decisions.

Only keep contributions that are actually demonstrated by the final experiments.

---

# 33. Related Work categories

Search and organize prior literature into these categories:

## A. Structured pruning of Transformer layers

Discuss methods that remove:
- complete layers
- blocks
- attention components
- feed-forward components

## B. LLM pruning

Discuss:
- weight pruning
- structured pruning
- layer pruning
- post-training pruning

## C. Multilingual LLMs

Discuss:
- multilingual representation
- cross-language transfer
- language-specific behavior

## D. Calibration data

Discuss how pruning methods depend on representative calibration distributions.

## E. Indic NLP

Discuss the importance of evaluating Hindi and Marathi rather than treating English as a universal proxy.

The paper's gap should emerge from the combination of these areas.

---

# 34. Methodology section

A clean methodology structure:

## A. Models

Describe:
- model names
- parameter counts
- number of layers
- hidden dimensions
- tokenizer
- model versions

## B. Data

Describe:
- source
- languages
- sampling
- calibration split
- evaluation split
- token budgets

## C. Layer Influence

Give the equation:

```text
I_l = 1 - mean_t cos(H_l,t^in, H_l,t^out)
```

where:
- `l` = layer
- `t` = token
- `H^in` = layer input
- `H^out` = layer output

## D. Ranking

Rank layers from lowest to highest influence.

## E. Structured Pruning

Remove the lowest-influence layers.

## F. Evaluation

Measure:
- PPL
- relative degradation
- rank agreement
- stability
- optional downstream performance

---

# 35. Threats to validity

This section is important for publication quality.

## Dataset bias

Wikipedia is not representative of every real-world language use case.

## Calibration size

Small calibration sets may produce unstable rankings.

## Metric limitation

Block Influence measures representation change, not causal functional importance.

## Tokenization confounding

Different languages may receive different tokenization behavior.

## Model coverage

Three models still do not represent all multilingual LLMs.

## Architecture effects

Different architectures may respond differently to deleting layers.

## Domain mismatch

Wikipedia calibration/evaluation may not predict chat, code, instruction-following, or specialized-domain behavior.

## Downstream task coverage

One downstream task per language is not enough to establish general task robustness.

---

# 36. What would make this more publishable?

The strongest upgrades are:

### 1. Multiple seeds

Repeat calibration selection.

### 2. Random baseline

Show that the proposed ranking is better than random selection.

### 3. Multiple models

Demonstrate that the observation is not Qwen-specific.

### 4. Multiple pruning ratios

2/4/6 layers is a start. A final study can use percentages such as 10%, 20%, 30%.

### 5. Token-budget matching

Make language calibration exposure comparable.

### 6. Downstream task

Measure a task in addition to PPL.

### 7. Alternative influence metric

Compare cosine-based influence with relative L2 change.

### 8. Calibration-size sweep

Determine how many multilingual samples are sufficient.

This last experiment could become a particularly useful practical contribution.

---

# 37. Stronger proposed experiment: calibration budget

A valuable extension is:

```text
Calibration budget per language

1
5
10
20
50
100
```

For each budget:

```text
English only
Hindi only
Marathi only
Balanced EN+HI+MR
```

Then measure:

```text
rank stability
Indic PPL degradation
```

The question becomes:

> How much multilingual calibration data is needed before pruning becomes language-robust?

This is more actionable than simply showing that languages differ.

---

# 38. Stronger proposed experiment: language-aware selection

The current experiment ranks layers separately.

A later method could define a multilingual objective.

For example:

```text
Mixed Influence(l)
=
mean(
    Influence_EN(l),
    Influence_HI(l),
    Influence_MR(l)
)
```

Then prune the layers with the smallest mixed influence.

This creates a simple language-aware pruning strategy.

Do not present this as a novel algorithm unless the literature review confirms that the formulation is genuinely novel.

---

# 39. Stronger proposed experiment: weighted calibration

A future method could use:

```text
I_weighted(l)
=
w_EN I_EN(l)
+
w_HI I_HI(l)
+
w_MR I_MR(l)
```

where:

```text
w_EN + w_HI + w_MR = 1
```

Different weights could represent deployment distributions.

For example:

```text
70% English
15% Hindi
15% Marathi
```

The paper could then study the trade-off between language-specific and deployment-aware pruning.

Again, check related literature before claiming novelty.

---

# 40. Important distinction: correlation versus causation

Suppose you observe:

```text
Marathi has higher tokens/word
and
Marathi has a different layer ranking
```

You cannot conclude:

> Higher tokenization caused the different layer ranking.

The correct statement is:

> The observed ranking differences were associated with differences in tokenizer token-per-word statistics.

To investigate causality, you would need controlled experiments.

---

# 41. What counts as a strong result?

A strong result is not necessarily:

> "Our method achieves the lowest perplexity."

For this project, the strongest evidence is a coherent pattern such as:

```text
1. Rankings differ across languages.
2. Differences are stable across calibration seeds.
3. English-calibrated pruning disproportionately affects Indic evaluation.
4. Language-matched calibration reduces the degradation.
5. Mixed calibration recovers much of the lost robustness.
6. The pattern appears across more than one model.
7. Influence-based selection beats random pruning.
```

The actual paper should only claim the parts supported by the data.

---

# 42. What counts as a negative result?

If the experiment finds:

```text
English ranking ≈ Hindi ranking ≈ Marathi ranking
```

and:

```text
English calibration ≈ multilingual calibration
```

that is still scientifically useful.

The conclusion would then be that, under the tested models, datasets, and pruning levels, language-dependent redundancy was not strongly observed.

Do not force the hypothesis to be true.

---

# 43. Recommended final research pipeline

```text
Phase 1
------
Qwen pilot
EN/HI/MR
20 calibration chunks
20 evaluation chunks
2/4/6 layers

        |
        v

Phase 2
------
Check:
- influence
- rank agreement
- PPL
- plots

        |
        v

Phase 3
------
Add:
- mixed calibration
- random pruning
- multiple seeds

        |
        v

Phase 4
------
Increase data
Match token budgets
Add downstream benchmark

        |
        v

Phase 5
------
Replicate on Llama and Gemma

        |
        v

Phase 6
------
Statistical analysis
Tables
Figures
Discussion
Limitations

        |
        v

Phase 7
------
IEEE manuscript
```

---

# 44. Recommended paper claim discipline

Use:

> "Our experiments indicate..."

instead of:

> "This proves..."

Use:

> "We observe language-dependent differences..."

instead of:

> "Language causes..."

Use:

> "Block Influence is used as a proxy..."

instead of:

> "Block Influence measures true importance."

Use:

> "Under the evaluated models and datasets..."

instead of:

> "Multilingual LLMs always..."

This makes the paper more scientifically defensible.

---

# 45. References to start the literature review

The following official/model resources are useful for describing the experimental models and data:

- Qwen2.5 model documentation/model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Llama 3.2 model documentation/model card: https://huggingface.co/meta-llama/Llama-3.2-1B
- Gemma 2 documentation/model card: https://huggingface.co/google/gemma-2-2b
- Wikimedia Wikipedia dataset: https://huggingface.co/datasets/wikimedia/wikipedia
- AI4Bharat IndicSentiment: https://huggingface.co/ai4bharat/IndicSentiment
- IndicGLUE: https://huggingface.co/datasets/ai4bharat/IndicGLUE

For the actual IEEE paper, these should be supplemented with peer-reviewed literature on:
- Transformer layer pruning
- structured LLM pruning
- multilingual model compression
- calibration-based pruning
- multilingual representation analysis
- Indic NLP evaluation

Do not treat model cards or dataset cards as substitutes for peer-reviewed related work.

---

# 46. What to do after running the notebook

Send the experiment outputs back.

At minimum:

```text
block_influence_by_language.csv
rank_agreement.csv
pruning_perplexity_results.csv
tokenizer_tokens_per_word.csv
ranking_stability.csv
```

and the plots.

The next research stage is:

```text
Raw results
    |
    v
Statistical interpretation
    |
    v
Findings
    |
    v
Results section
    |
    v
Discussion
    |
    v
Limitations
    |
    v
IEEE paper
```

Do not write the final numerical Results section before the experiments are actually completed.

---

# 47. Final conceptual summary

The entire research idea can be remembered as:

```text
QUESTION
Are redundant Transformer layers the same across languages?

        |
        v

MEASURE
How much does each layer change hidden states?

        |
        v

RANK
Which layers appear least influential?

        |
        v

COMPARE
Do English, Hindi, and Marathi rank them differently?

        |
        v

PRUNE
Remove 2, 4, or 6 selected layers.

        |
        v

TEST
Measure English/Hindi/Marathi perplexity.

        |
        v

CONTROL
Compare with random pruning and multiple seeds.

        |
        v

EXPLAIN
Check tokenization and calibration effects.

        |
        v

CONCLUDE
Determine whether pruning calibration should be language-aware.
```

The key scientific question is therefore not simply:

> "Can we prune layers?"

It is:

> **"Does the language used to decide which layers are redundant change which layers should be pruned and how safely pruning transfers across languages?"**
