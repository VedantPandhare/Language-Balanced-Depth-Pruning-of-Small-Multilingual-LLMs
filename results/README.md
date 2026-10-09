# Final v4 results

This is the canonical result bundle for the language-balanced Block Influence
pruning study. It contains one directory per evaluated model:

| Directory | Checkpoint |
| --- | --- |
| `qwen/` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `llama/` | `meta-llama/Llama-3.2-1B` |
| `gemma/` | `google/gemma-2-2b` |

Each model directory includes its immutable repository revision in
`experiment_config.json`, along with article identifiers, influence scores,
pruning sets, primary and random-control results, confidence-interval
summaries, figures, and restartable NLL caches.

Legacy v3 outputs and duplicate v4 result trees are intentionally excluded.
