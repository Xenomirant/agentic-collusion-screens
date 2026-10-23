# Autoresearch loop: who-colludes recovery at scale

Goal: maximise mean precision@K of planted coalition pairs recovered by
`fathon.tensor.scale.detect` over the 24 simulated markets in
`data/planted_cases.jsonl` (N = 5k / 30k agents sharing a rank-4 / rank-12
common mode, copy pairs, copy triads and XOR triads, coupling 0.25 / 0.45,
T = 2048). Harness: `benchmark.py` (read-only during iteration; prints
`metric: <float>`). Regression gate: `cd fathon && pytest tests -q`, must keep
`.regression-baseline` passes. One change per iteration, committed before
verification, reverted when the metric does not improve.

Corpus note: ground truth requires planted coalitions, so the corpus is
simulated (user-approved exception to the real-data rule). Real transcripts
are used separately (E4, E8).
