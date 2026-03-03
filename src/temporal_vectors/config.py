"""Centralised constants and paths for the TemporalVectors project.

Every script and notebook should import from this module rather than
hard-coding magic numbers.
"""

from pathlib import Path

# ── Model ────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-3B"
HIDDEN_SIZE = 3072
NUM_LAYERS = 28
TARGET_LAYERS = [7, 14, 21, 27]  # quarter, half, three-quarter, final

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
OUTPUTS = PROJECT_ROOT / "outputs"
HIDDEN_DIR = OUTPUTS / "hidden_states"
VECTOR_DIR = OUTPUTS / "vectors"
RESULTS_DIR = OUTPUTS / "results"
FIGURES_DIR = OUTPUTS / "figures"

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ── Dataset targets ──────────────────────────────────────────────
MIN_NATURAL_PAIRS = 500
MIN_SYNTHETIC_PAIRS = 300
MIN_CONTROL_PAIRS = 200

# ── Extraction ───────────────────────────────────────────────────
EXTRACTION_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512

# ── Wikipedia ────────────────────────────────────────────────────
WIKI_RATE_LIMIT_SECONDS = 1.0
WIKI_TARGET_YEARS = [2020, 2021, 2022, 2023, 2024]

# ── Forecasting ──────────────────────────────────────────────────
FORECAST_ALPHA_RANGE = [0.5, 1.0, 1.5, 2.0]
