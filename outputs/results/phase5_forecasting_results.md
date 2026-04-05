# Phase 5 — Forecasting Results

## Method

h_forecast = h_old + alpha * gamma (scaled by mean delta norm)
Direction source: synthetic temporal direction
Metric: cosine similarity between h_forecast and h_new (higher = better)

## Synthetic pairs (direction from synthetic)

| Layer | Identity | Random | a=0.5 | a=1.0 | a=1.5 | a=2.0 |
|-------|----------|--------|-------|-------|-------|-------|
| 7     | 0.8688   | 0.7794 | **0.8843** | 0.8471 | 0.7727 | 0.6855 |
| 14    | 0.9005   | 0.8280 | **0.9101** | 0.8769 | 0.8077 | 0.7197 |
| 21    | 0.9366   | 0.8856 | **0.9431** | 0.9216 | 0.8745 | 0.8100 |
| 27    | 0.9540   | 0.9133 | **0.9594** | 0.9444 | 0.9127 | 0.8701 |

## Natural pairs (direction from synthetic)

| Layer | Identity | Random | a=0.5 | a=1.0 | a=1.5 | a=2.0 |
|-------|----------|--------|-------|-------|-------|-------|
| 7     | **0.7540** | 0.6315 | 0.7137 | 0.6215 | 0.5208 | 0.4339 |
| 14    | **0.7915** | 0.6821 | 0.7545 | 0.6674 | 0.5664 | 0.4747 |
| 21    | **0.8352** | 0.7340 | 0.8057 | 0.7333 | 0.6458 | 0.5620 |
| 27    | **0.8718** | 0.7855 | 0.8540 | 0.8040 | 0.7415 | 0.6792 |

## Control pairs (direction from synthetic)

| Layer | Identity | Random | a=0.5 | a=1.0 | a=1.5 | a=2.0 |
|-------|----------|--------|-------|-------|-------|-------|
| 7     | **0.9757** | 0.9550 | 0.9703 | 0.9548 | 0.9305 | 0.8994 |
| 14    | **0.9777** | 0.9594 | 0.9728 | 0.9584 | 0.9354 | 0.9048 |
| 21    | **0.9796** | 0.9625 | 0.9754 | 0.9627 | 0.9420 | 0.9145 |
| 27    | **0.9855** | 0.9727 | 0.9830 | 0.9745 | 0.9609 | 0.9429 |

## Synthetic domain breakdown (layer 27, a=0.5)

| Domain     | Cosine |
|------------|--------|
| science    | 0.9727 |
| economics  | 0.9711 |
| sports     | 0.9670 |
| geography  | 0.9640 |
| politics   | 0.9516 |
| technology | 0.9248 |

## Key findings

- Temporal vector improves forecasting for synthetic pairs (beats identity at all layers, best a=0.5)
- Both temporal and identity massively beat random baseline — confirms the direction captures real structure
- Identity wins for natural pairs — synthetic temporal direction doesn't generalize to noisy Wikipedia diffs
- Identity wins for control pairs — expected, temporal direction shouldn't help non-temporal changes
- Best alpha = 0.5 everywhere — temporal shifts are smaller than mean delta magnitude
- Improvement is modest but consistent: ~0.5% absolute improvement on synthetic
- Later layers show higher baseline cosine (old and new are more similar at deeper layers)
