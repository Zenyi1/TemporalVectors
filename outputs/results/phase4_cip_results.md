# Phase 4 — Causal Inner Product Results

## CIP vs Euclidean Parallelism (using synthetic temporal direction)

| Layer | Synthetic EUC | Synthetic CIP | Natural EUC | Natural CIP | Control EUC | Control CIP |
|-------|---------------|---------------|-------------|-------------|-------------|-------------|
| 7     | 0.377         | 0.356         | 0.007       | 0.006       | -0.004      | 0.000       |
| 14    | 0.368         | 0.352         | 0.007       | 0.006       | 0.004       | 0.022       |
| 21    | 0.372         | 0.366         | 0.007       | 0.008       | 0.018       | 0.013       |
| 27    | 0.348         | 0.340         | 0.006       | 0.012       | 0.038       | 0.029       |

## Orthogonality (temporal vs control direction)

| Layer | EUC cosine | CIP cosine | Pass (<0.2) |
|-------|------------|------------|-------------|
| 7     | -0.016     | -0.009     | yes         |
| 14    | 0.006      | 0.093      | yes         |
| 21    | 0.059      | 0.037      | yes         |
| 27    | 0.148      | 0.122      | yes         |

## Synthetic CIP parallelism by domain

| Domain     | L7    | L14    | L21   | L27   |
|------------|-------|--------|-------|-------|
| economics  | 0.490 | 0.508  | 0.483 | 0.481 |
| science    | 0.495 | 0.540  | 0.460 | 0.460 |
| sports     | 0.397 | 0.391  | 0.351 | 0.286 |
| politics   | 0.337 | 0.338  | 0.356 | 0.338 |
| geography  | 0.074 | -0.020 | 0.159 | 0.051 |
| technology | 0.051 | -0.097 | 0.108 | 0.052 |

## Key findings

- CIP parallelism closely matches Euclidean — temporal info is distributed, not concentrated in prediction-critical dimensions
- Orthogonality passes all layers — temporal and non-temporal directions are causally separable
- Domain ranking consistent across both metrics — economics/science strongest, geography/technology weakest
- Natural pairs remain near zero under CIP — content noise is not a CIP-fixable problem
