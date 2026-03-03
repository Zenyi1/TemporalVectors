#!/usr/bin/env python
"""Create all project directories and verify the environment.

Usage:
    python scripts/setup_project.py
"""

import importlib
import sys
from pathlib import Path

# ── Directories to ensure exist ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "src/temporal_vectors",
    "src/temporal_vectors/data",
    "src/temporal_vectors/models",
    "src/temporal_vectors/analysis",
    "src/temporal_vectors/evaluation",
    "src/temporal_vectors/visualisation",
    "src/temporal_vectors/utils",
    "notebooks",
    "scripts",
    "configs",
    "data/raw",
    "data/processed",
    "data/cache",
    "outputs/hidden_states",
    "outputs/vectors",
    "outputs/results",
    "outputs/figures",
    "tests",
    "thesis/figures",
]


def create_directories() -> None:
    """Create all required project directories."""
    print("Creating directories...")
    for d in REQUIRED_DIRS:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
    print(f"  {len(REQUIRED_DIRS)} directories verified.")


def check_package_import() -> bool:
    """Check that temporal_vectors is importable."""
    try:
        importlib.import_module("temporal_vectors.config")
        from temporal_vectors.config import PROJECT_ROOT as cfg_root

        print(f"  temporal_vectors.config imports OK")
        print(f"  PROJECT_ROOT = {cfg_root}")
        return True
    except ImportError as e:
        print(f"  FAIL: cannot import temporal_vectors.config: {e}")
        print("  Run: pip install -e .")
        return False


def check_dependencies() -> list[str]:
    """Check that critical dependencies are installed."""
    deps = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sklearn": "scikit-learn",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
        "mwclient": "mwclient",
        "accelerate": "Accelerate",
    }
    missing = []
    for module, name in deps.items():
        try:
            importlib.import_module(module)
            print(f"  {name:.<30s} OK")
        except ImportError:
            print(f"  {name:.<30s} MISSING")
            missing.append(name)
    return missing


def check_gpu() -> None:
    """Report GPU availability and specs."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("  GPU: not available (CPU only)")
    except ImportError:
        print("  GPU: cannot check (torch not installed)")


def main() -> None:
    print(f"TemporalVectors Setup")
    print(f"{'=' * 50}")
    print(f"Project root: {PROJECT_ROOT}\n")

    # 1. Directories
    create_directories()
    print()

    # 2. Package import
    print("Checking package import...")
    pkg_ok = check_package_import()
    print()

    # 3. Dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    print()

    # 4. GPU
    print("Checking GPU...")
    check_gpu()
    print()

    # 5. Summary
    print(f"{'=' * 50}")
    if not missing and pkg_ok:
        print("All checks passed. Ready to run experiments.")
    else:
        issues = []
        if not pkg_ok:
            issues.append("package not importable (run: pip install -e .)")
        if missing:
            issues.append(f"missing deps: {', '.join(missing)}")
        print(f"Issues found: {'; '.join(issues)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
