# TemporalVectors -- Project Guide

> **Linear Representations for Temporal Knowledge Forecasting in Large Language Models**
> Author: Carlos Zenyi Gomez Aryoshi (52319406) -- University of Aberdeen

## Core Hypothesis

Temporal concepts (year-to-year knowledge shifts, tense changes, knowledge staleness) are
represented as linear directions in LLM hidden-state spaces. Adding a learned "temporal
vector" to a historical representation can approximate the future knowledge state better
than baselines.

## Four Objectives

1. **Identify temporal vectors** -- discover linear directions for temporal shifts.
2. **Test linearity & compositionality** -- parallelism across time steps, composition with topic vectors.
3. **Evaluate forecasting** -- temporal-vector arithmetic vs. baselines.
4. **Validate causal separability** -- temporal shifts preserve non-temporal semantics (causal inner product).

## Primary Model

```
meta-llama/Llama-3.2-3B
  hidden_size  = 3072
  num_layers   = 28
  vocab_size   = 128256
```

Scale up to 7B/13B only if compute permits and 3B results are promising.

## Key Reference

Park, K., Choe, Y.J., and Veitch, V. *The Linear Representation Hypothesis and the
Geometry of Large Language Models.* ICML, 2024.

---

## Repository Structure

```
TemporalVectros/
├── CLAUDE.md                          # this file
├── pyproject.toml                     # project metadata & dependencies
├── setup.cfg                          # optional legacy config
├── environment.yml                    # conda env spec
│
├── src/
│   └── temporal_vectors/
│       ├── __init__.py
│       ├── config.py                  # centralised constants & paths
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── wikipedia_fetcher.py   # mwclient-based revision fetcher
│       │   ├── pair_builder.py        # counterfactual pair construction
│       │   ├── synthetic_pairs.py     # template-based synthetic pairs
│       │   └── dataset.py             # torch Dataset / DataLoader helpers
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── loader.py              # model & tokenizer loading
│       │   └── hooks.py               # forward-hook utilities for steering
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── hidden_states.py       # HiddenStateExtractor class
│       │   ├── temporal_vectors.py    # difference vectors & direction estimation
│       │   ├── linearity_tests.py     # parallelism, additivity, scaling, cross-domain
│       │   └── causal_inner_product.py # CausalInnerProduct class
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── forecasting.py         # vector-arithmetic forecasting
│       │   ├── baselines.py           # cosine-decay, random-direction, fine-tune probe
│       │   ├── probes.py              # linear probe accuracy
│       │   └── steering.py            # activation-steering experiments
│       │
│       ├── visualisation/
│       │   ├── __init__.py
│       │   ├── figures.py             # individual figure generators
│       │   └── style.py               # shared matplotlib rc & palette
│       │
│       └── utils/
│           ├── __init__.py
│           ├── reproducibility.py     # seed_everything, config hashing
│           ├── io.py                  # JSONL read/write, checkpoint helpers
│           └── stats.py              # bootstrap CI, permutation tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hidden_state_inspection.ipynb
│   ├── 03_temporal_vector_analysis.ipynb
│   ├── 04_causal_inner_product.ipynb
│   ├── 05_forecasting_results.ipynb
│   └── 06_final_figures.ipynb
│
├── scripts/
│   ├── setup_project.py               # create dirs, verify env
│   ├── fetch_wikipedia_pairs.py        # CLI: build natural pair dataset
│   ├── generate_synthetic_pairs.py     # CLI: build synthetic + control pairs
│   ├── extract_hidden_states.py        # CLI: batch extraction
│   ├── compute_temporal_vectors.py     # CLI: direction estimation
│   ├── run_linearity_tests.py          # CLI: 4 linearity tests
│   ├── compute_cip.py                  # CLI: causal inner product
│   ├── run_forecasting.py              # CLI: forecasting + baselines
│   ├── run_analysis.py                 # CLI: ablations, decomposition
│   └── generate_figures.py             # CLI: all thesis figures
│
├── configs/
│   ├── default.yaml                    # default hyperparams
│   └── debug.yaml                      # small-scale quick-run config
│
├── data/
│   ├── raw/                            # fetched Wikipedia revisions (gitignored)
│   ├── processed/                      # JSONL counterfactual pairs
│   └── cache/                          # HF model cache, embeddings cache
│
├── outputs/
│   ├── hidden_states/                  # .pt tensor files per layer
│   ├── vectors/                        # estimated temporal directions
│   ├── results/                        # metrics JSON / CSV
│   └── figures/                        # publication-ready PNGs / PDFs
│
├── tests/
│   ├── test_data.py
│   ├── test_hidden_states.py
│   ├── test_temporal_vectors.py
│   ├── test_cip.py
│   └── test_forecasting.py
│
├── thesis/
│   ├── main.tex
│   └── figures/                        # symlink or copy from outputs/figures
│
└── .gitignore
```

---

## Environment Setup

### Option A -- Conda (recommended)

Create `environment.yml`:

```yaml
name: temporal-vectors
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.2
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1
  - numpy>=1.24
  - scipy>=1.11
  - scikit-learn>=1.3
  - pandas>=2.0
  - matplotlib>=3.8
  - seaborn>=0.13
  - jupyter
  - ipykernel
  - tqdm
  - pyyaml
  - pip
  - pip:
    - transformers>=4.38
    - accelerate>=0.27
    - datasets>=2.17
    - mwclient>=0.10
    - sentencepiece
    - protobuf
    - plotly>=5.18
    - nbstripout
    - black
    - ruff
    - pytest
    - safetensors
```

```bash
conda env create -f environment.yml
conda activate temporal-vectors
pip install -e .            # install src/ as editable package
nbstripout --install        # auto-strip notebook outputs on commit
```

### Option B -- venv fallback

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install torch==2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt    # generate from environment.yml pip section
pip install -e .
```

### Model Download

LLaMA 3.2 is a gated model. You need:

1. A HuggingFace account with an access token.
2. Accepted licence at the model page for `meta-llama/Llama-3.2-3B`.
3. Login:

```bash
huggingface-cli login          # paste your HF_TOKEN
```

4. The first run of model loading will download weights (~6 GB) to `data/cache/` (or `HF_HOME`).

### Verification

```bash
python -c "
import torch, transformers
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB')
print('Transformers:', transformers.__version__)
"
```

---

## Configuration -- `src/temporal_vectors/config.py`

All magic numbers live here. Every script and notebook imports from this module.

```python
from pathlib import Path

# ── Model ────────────────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Llama-3.2-3B"
HIDDEN_SIZE  = 3072
NUM_LAYERS   = 28
TARGET_LAYERS = [7, 14, 21, 27]   # quarter, half, three-quarter, final

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_CACHE   = PROJECT_ROOT / "data" / "cache"
OUTPUTS      = PROJECT_ROOT / "outputs"
HIDDEN_DIR   = OUTPUTS / "hidden_states"
VECTOR_DIR   = OUTPUTS / "vectors"
RESULTS_DIR  = OUTPUTS / "results"
FIGURES_DIR  = OUTPUTS / "figures"

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ── Dataset targets ──────────────────────────────────────────────
MIN_NATURAL_PAIRS   = 500
MIN_SYNTHETIC_PAIRS = 300
MIN_CONTROL_PAIRS   = 200

# ── Extraction ───────────────────────────────────────────────────
EXTRACTION_BATCH_SIZE = 8
MAX_SEQ_LENGTH        = 512

# ── Wikipedia ────────────────────────────────────────────────────
WIKI_RATE_LIMIT_SECONDS = 1.0
WIKI_TARGET_YEARS       = [2020, 2021, 2022, 2023, 2024]

# ── Forecasting ──────────────────────────────────────────────────
FORECAST_ALPHA_RANGE = [0.5, 1.0, 1.5, 2.0]
```

---

## Experiment Phases

### Phase 0 -- Project Setup (W1-2)

**Goal:** Runnable skeleton -- environment, model loads, smoke test passes.

**Steps:**
1. Create every directory in the repo structure above.
2. Initialise git repo; add `.gitignore` (see Conventions below).
3. Create `environment.yml` and install.
4. Write `config.py` with all constants.
5. Write `models/loader.py`:
   ```python
   def load_model_and_tokenizer(model_name=None, device_map="auto", cache_dir=None):
       """Return (model, tokenizer) in eval mode, no grad."""
   ```
6. Smoke-test: load model, tokenize one sentence, run forward pass, print hidden-state shape.

**Inputs:** None (bootstrapping).
**Outputs:** Working env, model loads, `outputs/` dirs exist.

**Success criteria:**
- `conda activate temporal-vectors && python -c "from temporal_vectors.config import *; print(PROJECT_ROOT)"` prints the correct path.
- `python scripts/setup_project.py` creates all dirs without error.
- Smoke-test forward pass returns tensor shape `(1, seq_len, 3072)`.

---

### Phase 1 -- Dataset Construction (W2-4)

**Goal:** Build three counterfactual-pair datasets in JSONL format.

#### 1a. Natural Pairs (Wikipedia revisions)

**Implementation -- `data/wikipedia_fetcher.py`:**
- Use `mwclient` to connect to English Wikipedia.
- For each target article, fetch revisions at two timestamps from `WIKI_TARGET_YEARS`.
- Extract lead section (first paragraph) from each revision.
- Diff the two versions to isolate changed sentences.

**Implementation -- `data/pair_builder.py`:**
- Take (old_sentence, new_sentence, article, year_old, year_new) tuples.
- Filter: both sentences must be >10 tokens, <200 tokens; change must be factual (not formatting).
- Write JSONL with schema:

```json
{
  "id": "nat_0001",
  "type": "natural",
  "domain": "politics",
  "text_old": "Biden will be the 46th president.",
  "text_new": "Biden is the 46th president.",
  "year_old": 2020,
  "year_new": 2021,
  "article": "Joe_Biden",
  "metadata": {"change_type": "tense", "revision_ids": [12345, 67890]}
}
```

**Target domains** (aim for balance): politics, technology, science, sports, geography, economics.

**Script:** `scripts/fetch_wikipedia_pairs.py --config configs/default.yaml`

**Rate limiting:** 1 request/second (`WIKI_RATE_LIMIT_SECONDS`), cache raw HTML in `data/raw/`.

#### 1b. Synthetic Pairs

**Implementation -- `data/synthetic_pairs.py`:**
- Template-based generation for controlled experiments:
  - **Tense shifts:** "X is happening" / "X happened" (with entity slot-filling).
  - **Year-stamped facts:** "As of {year_old}, the population of {city} is {val_old}" / "As of {year_new} ...".
  - **Status changes:** "X is the current Y" / "X is the former Y".
- Each template produces a pair; metadata records template ID and slot values.

**Schema:** Same JSONL as above but `"type": "synthetic"`.

#### 1c. Control Pairs

- Pairs that change a **non-temporal** attribute (e.g., spelling correction, synonym swap).
- Used to verify that temporal vectors are *not* activated by non-temporal edits.

**Schema:** Same JSONL, `"type": "control"`.

**Inputs:** Wikipedia API access, template files.
**Outputs:**
- `data/processed/natural_pairs.jsonl` (>= 500 pairs)
- `data/processed/synthetic_pairs.jsonl` (>= 300 pairs)
- `data/processed/control_pairs.jsonl` (>= 200 pairs)

**Success criteria:**
- Each file meets minimum count.
- Pairs pass schema validation (all required fields present, types correct).
- Domain distribution: no single domain > 40% of natural pairs.
- Manual spot-check of 20 random pairs confirms quality.

---

### Phase 2 -- Hidden State Extraction (W4-6)

**Goal:** Extract per-layer hidden states for every sentence in the dataset.

**Implementation -- `analysis/hidden_states.py`:**

```python
class HiddenStateExtractor:
    """Extract hidden states from specified layers for a batch of texts.

    Usage:
        extractor = HiddenStateExtractor(model, tokenizer, layers=[7,14,21,27])
        states = extractor.extract("The president is Biden.")
        # states: dict[int, Tensor]  -- layer -> (1, hidden_size)
    """

    def __init__(self, model, tokenizer, layers=None, pooling="last_token"):
        ...

    def extract(self, text: str) -> dict[int, torch.Tensor]:
        """Single text -> {layer: pooled_hidden_state}."""

    def extract_batch(self, texts: list[str]) -> dict[int, torch.Tensor]:
        """Batch of texts -> {layer: (batch, hidden_size)}."""

    def extract_dataset(self, pairs_path: Path, output_dir: Path):
        """Process full JSONL; save .pt files per layer."""
```

**Key details:**
- **Pooling:** last-token hidden state (standard for causal LMs; the final token attends to all prior tokens).
- **Unembedding space:** optionally project `h` through the model's `lm_head` weight to get logit-space vectors (useful for CIP vocabulary covariance).
- **Memory management:** process in batches of `EXTRACTION_BATCH_SIZE`; move tensors to CPU after extraction; call `torch.cuda.empty_cache()` between batches.
- **Checkpointing:** save every 100 batches; resume from last checkpoint on restart.
- **Dtype:** use `torch.float32` for hidden states (not bfloat16) to preserve numerical precision for downstream linear algebra.

**Script:** `scripts/extract_hidden_states.py --pairs data/processed/natural_pairs.jsonl --layers 7 14 21 27`

**Inputs:** JSONL pair files, loaded model.
**Outputs:** `outputs/hidden_states/{pair_type}/layer_{L}.pt` -- shape `(N, hidden_size)` per layer.

**Success criteria:**
- Output tensors have correct shapes: `(N_pairs * 2, 3072)` per layer (both old and new texts).
- No NaN/Inf values in any tensor.
- Total extraction completes without OOM (monitor with `nvidia-smi`).

---

### Phase 3 -- Temporal Vector Extraction (W4-6)

**Goal:** Compute difference vectors and estimate a stable temporal direction; validate linearity.

**Implementation -- `analysis/temporal_vectors.py`:**

```python
def compute_difference_vectors(
    h_old: torch.Tensor,   # (N, D)
    h_new: torch.Tensor,   # (N, D)
) -> torch.Tensor:
    """Return delta = h_new - h_old, shape (N, D)."""
    return h_new - h_old


def estimate_temporal_direction(
    deltas: torch.Tensor,  # (N, D)
    method: str = "mean",  # "mean" | "pca"
) -> torch.Tensor:
    """Estimate a single temporal direction (unit vector) from many deltas.

    - "mean": normalise the mean delta.
    - "pca": first principal component of deltas.
    Returns shape (D,).
    """


def leave_one_out_stability(deltas: torch.Tensor) -> dict:
    """Leave-one-out cross-validation of direction stability.
    Returns mean cosine similarity between held-out direction and full direction."""
```

**Four linearity tests -- `analysis/linearity_tests.py`:**

| Test | What it measures | Pass threshold |
|------|-----------------|----------------|
| **Parallelism** | cos(direction_2020_2021, direction_2022_2023) | > 0.7 |
| **Additivity** | ‖(d_20_21 + d_21_22) - d_20_22‖ / ‖d_20_22‖ | < 0.3 |
| **Scaling** | cos(d_1yr, 0.5 * d_2yr) after normalisation | > 0.7 |
| **Cross-domain** | cos(d_politics, d_science) for same year gap | > 0.5 |

**Script:** `scripts/compute_temporal_vectors.py && scripts/run_linearity_tests.py`

**Inputs:** Hidden-state `.pt` files from Phase 2.
**Outputs:**
- `outputs/vectors/temporal_direction_layer{L}.pt` -- shape `(D,)` per layer.
- `outputs/results/linearity_tests.json` -- test names, scores, pass/fail.

**Success criteria:**
- Leave-one-out cosine similarity > 0.6 (direction is stable).
- At least 3 of 4 linearity tests pass at the best layer.

---

### Phase 4 -- Causal Inner Product (W6-8)

**Goal:** Build the CIP framework to measure temporal vectors in a semantically principled way.

**Implementation -- `analysis/causal_inner_product.py`:**

```python
class CausalInnerProduct:
    """Causal inner product following Park et al. (2024).

    The CIP accounts for vocabulary covariance so that concept
    measurements are not confounded by correlated features.

    Usage:
        cip = CausalInnerProduct(model)
        cip.fit(hidden_states)               # estimate Sigma_V
        cos = cip.cosine(gamma, delta)       # CIP cosine similarity
        proj = cip.project(h, gamma)         # project h onto gamma
    """

    def __init__(self, model, regularisation: float = 1e-4):
        self.W_U = model.lm_head.weight.detach().float()  # (vocab, D)
        self.reg = regularisation
        self.Sigma_V_inv = None

    def fit(self, hidden_states: torch.Tensor):
        """Estimate vocabulary covariance and compute regularised inverse.

        Sigma_V = W_U^T @ W_U  (D x D)
        Sigma_V_inv = (Sigma_V + reg * I)^{-1}
        """

    def inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """CIP: <u, v>_CIP = u^T @ Sigma_V_inv @ v"""

    def cosine(self, u: torch.Tensor, v: torch.Tensor) -> float:
        """CIP cosine similarity."""

    def norm(self, u: torch.Tensor) -> float:
        """CIP norm: sqrt(<u, u>_CIP)."""

    def project(self, h: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """CIP projection of h onto direction."""
```

**Orthogonality tests:**
- Compute CIP cosine between temporal direction and control concept directions (e.g., gender, language).
- Temporal direction should be near-orthogonal to controls: |CIP_cos| < 0.2.

**Script:** `scripts/compute_cip.py --layers 7 14 21 27`

**Inputs:** Hidden-state tensors, temporal direction vectors, model weights.
**Outputs:**
- `outputs/results/cip_metrics.json` -- CIP cosine/norm/projection for temporal vs. control directions.
- `outputs/results/orthogonality_tests.json`

**Success criteria:**
- CIP computation completes without numerical instability (condition number of Sigma_V < 1e8).
- Temporal direction has |CIP_cos| < 0.2 with at least 2 control directions.
- CIP cosine between temporal vectors from different year-gaps is > 0.5 (confirming they measure the same concept).

---

### Phase 5 -- Forecasting (W8-10)

**Goal:** Test whether temporal-vector arithmetic predicts future knowledge states.

#### 5a. Basic Forecasting

```python
def forecast(h_t: torch.Tensor, gamma: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """h_forecast = h(T) + alpha * gamma"""
    return h_t + alpha * gamma
```

Evaluate: `cosine_similarity(h_forecast, h_actual_future)` across held-out pairs.
Sweep `alpha` over `FORECAST_ALPHA_RANGE`.

#### 5b. Baselines -- `evaluation/baselines.py`

| Baseline | Description |
|----------|-------------|
| **Cosine decay** | Predict h_future = h_t (identity; temporal shift = 0) |
| **Random direction** | h_t + alpha * random_unit_vector |
| **Fine-tune probe** | Linear probe trained to predict future hidden state from past |

#### 5c. Probe Accuracy -- `evaluation/probes.py`

Train a linear classifier: given h, predict the year/period label. Report accuracy as a sanity check that hidden states encode temporal information at all.

#### 5d. Compositional Forecasting

```
h_forecast = h(topic_T) + alpha * gamma_temporal + beta * gamma_topic
```

Test whether adding both a temporal and a domain-specific vector improves prediction.

#### 5e. Multi-step Forecasting

Apply temporal vector repeatedly: `h(T) + k * gamma` for k = 1, 2, 3. Measure degradation.

#### 5f. Steering Experiments -- `evaluation/steering.py`

```python
def steering_hook(module, input, output, direction, alpha):
    """Forward hook that adds alpha * direction to hidden states."""
    output[0] = output[0] + alpha * direction.to(output[0].device)
    return output
```

Register hook on a target layer, generate text, evaluate whether output reflects the temporal shift.

**Script:** `scripts/run_forecasting.py --config configs/default.yaml`

**Inputs:** Hidden states, temporal direction, held-out test pairs.
**Outputs:**
- `outputs/results/forecasting_metrics.json` -- per-method cosine similarities, probe accuracy.
- `outputs/results/steering_examples.json` -- generated text samples before/after steering.

**Success criteria:**
- Temporal-vector forecast achieves higher cosine similarity to ground truth than all 3 baselines (statistically significant, p < 0.05, bootstrap test).
- Probe accuracy > 60% (above chance for multi-class year classification).
- Steering produces qualitatively coherent temporal shifts in at least 70% of examples.

---

### Phase 6 -- Analysis & Ablations (W10-12)

**Goal:** Deep analysis of what works, what doesn't, and why.

#### 6a. Linear/Nonlinear Decomposition

For each pair, decompose `delta = proj(delta, gamma) + residual`. Report:
- Fraction of variance explained by linear component.
- Characterise residual: is it structured or noise?

#### 6b. Layer-wise Analysis

Run all metrics at every `TARGET_LAYER`. Plot metric vs. layer to find where temporal information is most linearly represented.

#### 6c. Domain Breakdown

Report all metrics broken down by domain (politics, tech, science, etc.). Identify which domains are most/least linear.

#### 6d. Ablation Studies

| Ablation | What varies | Expected insight |
|----------|-------------|------------------|
| Dataset size | 50, 100, 200, 500 pairs | Minimum data needed for stable direction |
| Pair type | Natural only, synthetic only, mixed | Which pair type contributes most |
| Pooling strategy | last_token vs. mean_pool | Best extraction method |
| Direction method | mean vs. PCA | Robustness of estimation |

#### 6e. Statistical Significance

- Bootstrap 95% confidence intervals on all key metrics.
- Permutation tests for linearity scores (null: random pairing of old/new).

#### 6f. CIP vs. Euclidean Comparison

Re-run orthogonality and forecasting metrics with standard dot product instead of CIP. Quantify improvement from using CIP.

**Script:** `scripts/run_analysis.py --full`

**Inputs:** All previous outputs.
**Outputs:**
- `outputs/results/decomposition.json`
- `outputs/results/layer_analysis.json`
- `outputs/results/domain_breakdown.json`
- `outputs/results/ablations.json`
- `outputs/results/significance_tests.json`
- `outputs/results/cip_vs_euclidean.json`

**Success criteria:**
- Linear component explains > 30% of variance in temporal deltas.
- Layer analysis identifies a clear peak layer (or range).
- Ablations produce monotonic trends (more data = better stability).
- All reported differences backed by p < 0.05.

---

### Phase 7 -- Thesis Figures (W11-13)

**Goal:** Generate all publication-quality figures for the thesis.

**Figure conventions:**
- Format: PDF (vector) for thesis, PNG (300 dpi) for presentations.
- Size: single-column = 3.5 in wide, double-column = 7 in wide.
- Font: matching thesis font (Computer Modern or Times).
- Colour palette: colourblind-safe (use `seaborn.color_palette("colorblind")`).
- All figures must have axis labels, legends, and titles.

**Key figures:**

| # | Figure | Source data |
|---|--------|-------------|
| 1 | Temporal direction consistency (cosine similarity heatmap across year-pairs) | `linearity_tests.json` |
| 2 | Parallelism scatter: d(t1->t2) vs d(t3->t4) projected onto top-2 PCs | `temporal_direction_*.pt` |
| 3 | Additivity residual bar chart per domain | `linearity_tests.json` |
| 4 | CIP orthogonality matrix (temporal vs. control concepts) | `orthogonality_tests.json` |
| 5 | Forecasting cosine similarity: method comparison bar chart with CIs | `forecasting_metrics.json` |
| 6 | Layer-wise metric line plot (x=layer, y=metric, one line per metric) | `layer_analysis.json` |
| 7 | Linear vs. nonlinear variance pie/bar by domain | `decomposition.json` |
| 8 | Ablation curves (dataset size vs. direction stability) | `ablations.json` |
| 9 | Steering examples: token probability shift heatmap | `steering_examples.json` |
| 10 | 2D PCA of temporal vectors coloured by year-gap | `temporal_direction_*.pt` |

**Script:** `scripts/generate_figures.py --all --format pdf png`

**Inputs:** All result JSON/CSV files.
**Outputs:** `outputs/figures/fig_{01..10}.{pdf,png}`

**Success criteria:**
- All 10 figures render without error.
- Figures match conventions (size, font, palette).
- No missing data points or empty panels.

---

## Data Pipeline Reference

### Wikipedia Fetch Pipeline

```
Article list (curated per domain)
    │
    ▼
mwclient.Site("en.wikipedia.org")
    │  page.revisions(start=ts1, end=ts2, prop="content|timestamp")
    │  Rate limit: 1 req/sec, retry on 429
    ▼
Raw wikitext cached in data/raw/{article}_{year}.txt
    │
    ▼
Parse: extract lead section (first paragraph before first ==heading==)
    │
    ▼
Diff: sentence-level diff (difflib.SequenceMatcher)
    │  Filter: keep only changed sentences with factual content
    ▼
JSONL pair (text_old, text_new, metadata)
    → data/processed/natural_pairs.jsonl
```

### Synthetic Generation Pipeline

```
Template bank (YAML or Python dicts)
    │  e.g., "As of {year}, the president of {country} is {name}."
    │
    ▼
Slot-filling from curated fact table
    │  (country, year) → name lookup
    ▼
JSONL pair (text_old, text_new, metadata)
    → data/processed/synthetic_pairs.jsonl
```

### Caching Strategy

- **Wikipedia raw text:** `data/raw/` -- persisted across runs; never re-fetch if file exists.
- **HuggingFace models:** `data/cache/` or `$HF_HOME` -- set `TRANSFORMERS_CACHE` env var.
- **Hidden states:** `outputs/hidden_states/` -- checkpointed; skip already-extracted pairs on restart.

---

## Key Implementation Details

### Hidden State Extraction Strategy

- **Last-token pooling:** For a causal LM, the last token's hidden state attends to the full sequence. This is the standard representation for the sentence.
- **Unembedding projection:** To analyse in vocabulary space, multiply `h @ W_U.T` where `W_U = model.lm_head.weight`. This maps the hidden state to logit space, which is needed for computing the vocabulary covariance in CIP.
- **Numerical precision:** Always extract in `float32`. The difference vectors and covariance matrices are sensitive to precision.

### Difference Vector Computation

```python
# For each counterfactual pair (old_text, new_text):
h_old = extractor.extract(old_text)[layer]   # (D,)
h_new = extractor.extract(new_text)[layer]   # (D,)
delta = h_new - h_old                        # (D,)

# Temporal direction = normalised mean of deltas:
gamma = deltas.mean(dim=0)                   # (D,)
gamma = gamma / gamma.norm()                 # unit vector
```

### CausalInnerProduct -- Full API

```python
cip = CausalInnerProduct(model, regularisation=1e-4)
cip.fit(hidden_states)  # hidden_states: (N, D) from training set

# Measure temporal concept strength in a hidden state:
strength = cip.project(h, gamma)             # scalar

# Compare two directions:
sim = cip.cosine(gamma_temporal, gamma_topic) # scalar in [-1, 1]

# CIP norm (magnitude of concept):
mag = cip.norm(gamma)                         # scalar >= 0
```

### Model Steering Hook

```python
import functools

def make_steering_hook(direction, alpha):
    def hook(module, input, output):
        # output is a tuple; hidden states are output[0]
        output[0][:, -1, :] += alpha * direction.to(output[0].device)
        return output
    return hook

# Register on a specific layer:
handle = model.model.layers[TARGET_LAYER].register_forward_hook(
    make_steering_hook(gamma, alpha=1.0)
)
# Generate text...
handle.remove()  # always clean up
```

---

## Conventions

### Code Style

- **Formatter:** `black` (line length 99).
- **Linter:** `ruff` (select = `["E", "F", "I", "W"]`).
- **Docstrings:** Google style.
- **Type hints:** required on all public functions.
- Run before every commit: `black src/ scripts/ tests/ && ruff check src/ scripts/ tests/`

### Naming

| Scope | Convention | Example |
|-------|-----------|---------|
| Files / modules | `snake_case` | `hidden_states.py` |
| Classes | `PascalCase` | `HiddenStateExtractor` |
| Functions / variables | `snake_case` | `compute_difference_vectors` |
| Constants | `UPPER_SNAKE` | `HIDDEN_SIZE` |
| Config keys (YAML) | `snake_case` | `extraction_batch_size` |

### Git

- **Conventional commits:** `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- **Branch naming:** `feat/phase-N-description`, `fix/issue-description`
- **.gitignore** must include:
  ```
  data/raw/
  data/cache/
  outputs/hidden_states/
  outputs/vectors/
  *.pt
  *.bin
  __pycache__/
  .venv/
  *.egg-info/
  .ipynb_checkpoints/
  ```

### Notebooks

- First cell: `%load_ext autoreload` / `%autoreload 2`.
- No heavy compute in notebooks -- call functions from `src/` package.
- Use `nbstripout` to keep notebooks clean in version control.
- Notebooks are for exploration and visualisation, not production code.

### Reproducibility

- Call `seed_everything(SEED)` at the start of every script:
  ```python
  def seed_everything(seed: int = 42):
      import random, numpy as np, torch
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
- Log config hash with every result file for traceability.

---

## Verification Checklist

### Phase 0 -- Setup
- [ ] `conda activate temporal-vectors` succeeds
- [ ] `python -c "from temporal_vectors.config import *"` works
- [ ] Model loads and forward pass returns shape `(1, seq_len, 3072)`
- [ ] All output directories exist

### Phase 1 -- Dataset
- [ ] `natural_pairs.jsonl` >= 500 pairs
- [ ] `synthetic_pairs.jsonl` >= 300 pairs
- [ ] `control_pairs.jsonl` >= 200 pairs
- [ ] Schema validation passes on all files
- [ ] No domain exceeds 40% of natural pairs
- [ ] Manual spot-check passes

### Phase 2 -- Hidden States
- [ ] `.pt` files exist for all target layers
- [ ] Tensor shapes correct: `(N*2, 3072)`
- [ ] No NaN or Inf values
- [ ] Extraction completed without OOM

### Phase 3 -- Temporal Vectors
- [ ] Direction vectors saved per layer
- [ ] Leave-one-out cosine > 0.6
- [ ] >= 3/4 linearity tests pass at best layer

### Phase 4 -- Causal Inner Product
- [ ] Sigma_V condition number < 1e8
- [ ] |CIP_cos(temporal, control)| < 0.2 for >= 2 controls
- [ ] CIP_cos between same-concept temporal vectors > 0.5

### Phase 5 -- Forecasting
- [ ] Temporal forecast beats all 3 baselines (p < 0.05)
- [ ] Probe accuracy > 60%
- [ ] Steering produces coherent shifts in >= 70% examples

### Phase 6 -- Analysis
- [ ] Linear component explains > 30% variance
- [ ] Clear peak layer identified
- [ ] Ablation trends are monotonic
- [ ] All differences have p < 0.05

### Phase 7 -- Figures
- [ ] All 10 figures render without error
- [ ] Figures match size/font/palette conventions
- [ ] No missing data or empty panels

---

## Quick Reference Commands

### Full Pipeline (production)

```bash
# Phase 0
python scripts/setup_project.py

# Phase 1
python scripts/fetch_wikipedia_pairs.py --config configs/default.yaml
python scripts/generate_synthetic_pairs.py --config configs/default.yaml

# Phase 2
python scripts/extract_hidden_states.py \
    --pairs data/processed/natural_pairs.jsonl \
    --layers 7 14 21 27

python scripts/extract_hidden_states.py \
    --pairs data/processed/synthetic_pairs.jsonl \
    --layers 7 14 21 27

python scripts/extract_hidden_states.py \
    --pairs data/processed/control_pairs.jsonl \
    --layers 7 14 21 27

# Phase 3
python scripts/compute_temporal_vectors.py --layers 7 14 21 27
python scripts/run_linearity_tests.py

# Phase 4
python scripts/compute_cip.py --layers 7 14 21 27

# Phase 5
python scripts/run_forecasting.py --config configs/default.yaml

# Phase 6
python scripts/run_analysis.py --full

# Phase 7
python scripts/generate_figures.py --all --format pdf png
```

### Debug Pipeline (fast iteration)

```bash
python scripts/fetch_wikipedia_pairs.py --config configs/debug.yaml --max-pairs 20
python scripts/extract_hidden_states.py \
    --pairs data/processed/natural_pairs.jsonl --layers 14 --max-pairs 20
python scripts/compute_temporal_vectors.py --layers 14
python scripts/run_linearity_tests.py --layers 14
python scripts/run_forecasting.py --config configs/debug.yaml
```

### Testing & Linting

```bash
pytest tests/ -v --tb=short
black src/ scripts/ tests/ --check
ruff check src/ scripts/ tests/
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| **CUDA OOM** during extraction | Batch too large or model in float32 | Reduce `EXTRACTION_BATCH_SIZE` to 4 or 2; load model in `bfloat16` (extraction tensors remain float32) |
| **HuggingFace gated model error** | Missing or expired token / licence not accepted | Run `huggingface-cli login`; accept licence at model page |
| **Wikipedia 429 rate limit** | Requests too fast | Increase `WIKI_RATE_LIMIT_SECONDS` to 2.0; add exponential backoff |
| **NaN in covariance matrix** | Numerical instability in Sigma_V inversion | Increase `regularisation` to 1e-3; verify hidden states are float32 |
| **Sigma_V condition number too high** | Near-singular covariance | Increase regularisation; consider PCA dimensionality reduction first |
| **Import errors for `temporal_vectors`** | Package not installed in editable mode | Run `pip install -e .` from project root |
| **Notebook kernel can't find package** | Wrong kernel selected | `python -m ipykernel install --user --name temporal-vectors` |
| **Git repo too large** | `.pt` or model files committed | Check `.gitignore` includes `*.pt`, `*.bin`, `data/cache/`, `outputs/hidden_states/` |
| **Inconsistent results across runs** | Missing seed | Ensure `seed_everything(SEED)` is called at script entry point |
| **Slow Wikipedia fetching** | Sequential requests + no cache | Verify cache in `data/raw/`; consider async fetching with `aiohttp` |
