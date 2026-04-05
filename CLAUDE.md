# TemporalVectors

> **Linear Representations for Temporal Knowledge Forecasting in Large Language Models**
> Carlos Zenyi Gomez Aryoshi (52319406) -- University of Aberdeen
> Reference: Park et al., *The Linear Representation Hypothesis and the Geometry of Large Language Models*, ICML 2024.

## Hypothesis

Temporal concepts (year-to-year knowledge shifts, tense changes, knowledge staleness) are represented as **linear directions** in LLM hidden-state spaces. Adding a learned "temporal vector" to a historical representation can approximate the future knowledge state better than baselines.

## Objectives

1. **Identify temporal vectors** -- linear directions for temporal shifts
2. **Test linearity & compositionality** -- parallelism across time steps, composition with topic vectors
3. **Evaluate forecasting** -- temporal-vector arithmetic vs. baselines
4. **Validate causal separability** -- temporal shifts preserve non-temporal semantics (causal inner product)

## Model

`meta-llama/Llama-3.2-3B` -- hidden_size=3072, num_layers=28, target layers=[7, 14, 21, 27]. Scale to 7B/13B only if 3B results are promising.

---

## What's Built

### Phase 0 -- Project Skeleton (done)

Core infrastructure is in place. All constants live in `config.py`; every script imports from there.

- `src/temporal_vectors/config.py` -- all paths, model params, thresholds
- `src/temporal_vectors/utils/` -- `seed_everything()`, JSONL I/O, checkpointing, bootstrap CI, permutation tests
- `src/temporal_vectors/models/loader.py` -- `load_model_and_tokenizer()` loads LLaMA in bfloat16, forces CUDA when available
- `src/temporal_vectors/models/hooks.py` -- `steer_model()` and `capture_hidden_states()` context managers, compatible with transformers 5.x
- `scripts/setup_project.py` -- run this to verify dirs + deps + GPU
- `configs/default.yaml` / `configs/debug.yaml` -- production vs fast-iteration settings

**How to set up:**
```bash
conda env create -f environment.yml && conda activate temporal-vectors
pip install -e .
huggingface-cli login   # for gated LLaMA access
python scripts/setup_project.py
```

### Phase 1 -- Dataset Construction (done)

Three counterfactual-pair datasets, all sharing a common JSONL schema:

```json
{"id": "nat_0001", "type": "natural", "domain": "politics",
 "text_old": "...", "text_new": "...", "year_old": 2020, "year_new": 2021,
 "article": "Joe_Biden", "metadata": {"change_type": "tense"}}
```

**Natural pairs** (`data/wikipedia_fetcher.py` + `data/pair_builder.py`):
- Fetches Wikipedia article revisions at two time points via mwclient
- Extracts lead sections, diffs at sentence level, filters for quality (10-200 tokens, 5-90% change ratio, no formatting-only edits)
- Classifies changes as tense/status/numerical/factual
- 20 curated articles per domain (politics, technology, science, sports, geography, economics)
- Cached in `data/raw/`; run: `python scripts/fetch_wikipedia_pairs.py --config configs/default.yaml`

**Synthetic pairs** (`data/synthetic_pairs.py`):
- Fact tables: 10 countries with leaders by year, 10 cities with populations, 10 tech status changes, 10 event progressions, 5 tense templates x 8 slot fills
- Run: `python scripts/generate_synthetic_pairs.py --config configs/default.yaml`

**Control pairs** (same module):
- Non-temporal changes: synonym swaps, British/American spelling variants
- Same year_old and year_new -- used to verify temporal vectors don't fire on irrelevant edits

**DataLoader** (`data/dataset.py`): `PairDataset.from_jsonl(path)` yields individual texts with pair_id/side/domain metadata, ready for batched extraction.

**Targets:** >= 500 natural, >= 300 synthetic, >= 200 control. No domain > 40% of natural pairs.

### Phase 2 -- Hidden State Extraction (done)

Per-layer hidden states extracted for every sentence in all three datasets using last-token pooling, float32 precision.

- `src/temporal_vectors/analysis/hidden_states.py` -- `HiddenStateExtractor` class with batch processing, padding-aware pooling, checkpointing every 100 batches
- `scripts/extract_hidden_states.py` -- CLI that accepts multiple pair files in one run
- Output in `outputs/hidden_states/{pair_type}/layer_{L}.pt`, saved as `{"tensor": Tensor, "metadata": dict}`

**Results:**

| Dataset | Texts | Shape per layer | Layers | Total size |
|---------|-------|-----------------|--------|------------|
| Natural | 11,172 | (11172, 3072) | 4 | 524 MB |
| Synthetic | 648 | (648, 3072) | 4 | 31 MB |
| Control | 530 | (530, 3072) | 4 | 25 MB |

All tensors validated: float32, no NaN, no Inf.

---

## What's Next

### Phase 3 -- Temporal Vector Extraction (next up)

Compute `delta = h_new - h_old` for each pair, estimate a stable temporal direction (mean or PCA of deltas). Four linearity tests: parallelism (>0.7), additivity (<0.3 residual), scaling (>0.7), cross-domain (>0.5). Leave-one-out stability >0.6.

### Phase 4 -- Causal Inner Product

`CausalInnerProduct` class using Park et al.'s framework. Vocabulary covariance `Sigma_V = W_U^T @ W_U`, regularised inverse for CIP cosine/norm/projection. Validates temporal direction is orthogonal to control concepts (|CIP_cos| < 0.2).

### Phase 5 -- Forecasting

`h_forecast = h(T) + alpha * gamma`. Sweep alpha over [0.5, 1.0, 1.5, 2.0]. Compare against baselines (identity, random direction, linear probe). Also: compositional forecasting (temporal + topic vectors), multi-step (`k * gamma`), and activation steering experiments.

### Phase 6 -- Analysis & Ablations

Linear/nonlinear variance decomposition, layer-wise analysis, domain breakdown, ablations (dataset size, pair type, pooling, direction method), CIP vs Euclidean comparison. All with bootstrap CIs and permutation tests (p < 0.05).

### Phase 7 -- Thesis Figures

10 publication-quality figures: direction consistency heatmap, parallelism scatter, additivity bars, CIP orthogonality matrix, forecasting comparison, layer-wise metrics, variance decomposition, ablation curves, steering heatmap, PCA of temporal vectors. PDF + PNG, colourblind-safe palette.

---

## Conventions

- **Style:** `black` (line length 99), `ruff`, Google docstrings, type hints on public functions
- **Naming:** snake_case files/functions, PascalCase classes, UPPER_SNAKE constants
- **Git:** short commit messages, no `feat:` prefix
- **Notebooks:** for exploration only; call `src/` functions, use autoreload
- **Reproducibility:** `seed_everything(42)` at every script entry point

## Quick Reference

```bash
# Full pipeline
python scripts/setup_project.py
python scripts/fetch_wikipedia_pairs.py --config configs/default.yaml
python scripts/generate_synthetic_pairs.py --config configs/default.yaml
python scripts/extract_hidden_states.py --pairs data/processed/natural_pairs.jsonl data/processed/synthetic_pairs.jsonl data/processed/control_pairs.jsonl --layers 7 14 21 27
python scripts/compute_temporal_vectors.py --layers 7 14 21 27
python scripts/run_linearity_tests.py
python scripts/compute_cip.py --layers 7 14 21 27
python scripts/run_forecasting.py --config configs/default.yaml
python scripts/run_analysis.py --full
python scripts/generate_figures.py --all --format pdf png

# Debug (fast)
python scripts/fetch_wikipedia_pairs.py --config configs/debug.yaml --max-pairs 20
python scripts/extract_hidden_states.py --pairs data/processed/natural_pairs.jsonl --layers 14 --max-pairs 20
python scripts/compute_temporal_vectors.py --layers 14
python scripts/run_forecasting.py --config configs/debug.yaml
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | Reduce `--batch-size` to 4 or 2 |
| Model on CPU (slow) | Ensure laptop is plugged in (dGPU disabled on battery); check `torch.cuda.is_available()` |
| HF gated model error | `python -c "from huggingface_hub import login; login()"` + accept licence on model page |
| Wikipedia 429 | Increase `WIKI_RATE_LIMIT_SECONDS` to 2.0 |
| NaN in covariance | Increase CIP regularisation to 1e-3; ensure float32 hidden states |
| Import errors | `pip install -e .` from project root |
| Wrong notebook kernel | `python -m ipykernel install --user --name temporal-vectors` |
