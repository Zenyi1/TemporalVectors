# Phase 3 — Temporal Vector Extraction Results

## Parallelism (cosine similarity with temporal direction)

| Layer | Natural | Synthetic | Control |
|-------|---------|-----------|---------|
| 7     | 0.030   | 0.377     | 0.209   |
| 14    | 0.032   | 0.368     | 0.211   |
| 21    | 0.030   | 0.372     | 0.251   |
| 27    | 0.029   | 0.348     | 0.242   |

## Additivity residual (lower = more linear)

| Layer | Natural | Synthetic | Control |
|-------|---------|-----------|---------|
| 7     | 0.990   | 0.805     | 0.936   |
| 14    | 0.984   | 0.794     | 0.940   |
| 21    | 0.986   | 0.814     | 0.856   |
| 27    | 0.981   | 0.832     | 0.856   |

## Scaling correlation (time gap vs projection magnitude)

| Layer | Natural | Synthetic | Control |
|-------|---------|-----------|---------|
| 7     | 0.115   | -0.532    | 0.000   |
| 14    | 0.087   | -0.476    | 0.000   |
| 21    | 0.082   | -0.499    | 0.000   |
| 27    | 0.099   | -0.510    | 0.000   |

## Leave-one-out stability

All datasets, all layers: 1.000

## Cross-domain parallelism (synthetic, strongest signal)

| Domain     | L7    | L14   | L21   | L27   |
|------------|-------|-------|-------|-------|
| economics  | 0.500 | 0.472 | 0.491 | 0.504 |
| science    | 0.524 | 0.534 | 0.506 | 0.514 |
| sports     | 0.423 | 0.315 | 0.357 | 0.298 |
| politics   | 0.359 | 0.374 | 0.368 | 0.341 |
| geography  | 0.090 | 0.071 | 0.092 | 0.022 |
| technology | 0.069 | 0.085 | 0.061 | 0.033 |

## Key findings

- Synthetic pairs show clear temporal signal (0.37 parallelism) vs control (0.21) — real separation
- Natural pairs near zero — content noise dominates temporal signal in Wikipedia diffs
- Economics and science domains have strongest temporal linearity (~0.50)
- Direction is highly stable under resampling (LOO = 1.0)
- Temporal signal roughly uniform across layers — not concentrated at specific depth
- Negative scaling in synthetic suggests template structure effect, not true anti-correlation
- Method: mean direction, all layers extracted at float32
