#!/usr/bin/env python
"""Extract hidden states from LLaMA for all counterfactual pair texts.

Usage:
    python scripts/extract_hidden_states.py --pairs data/processed/natural_pairs.jsonl --layers 7 14 21 27
    python scripts/extract_hidden_states.py --pairs data/processed/natural_pairs.jsonl --layers 14 --max-pairs 20
"""

import argparse
import logging
from pathlib import Path

from temporal_vectors.config import (
    EXTRACTION_BATCH_SIZE,
    HIDDEN_DIR,
    SEED,
    TARGET_LAYERS,
)
from temporal_vectors.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden states from LLaMA")
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to JSONL pair file",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=TARGET_LAYERS,
        help="Layer indices to extract from (default: 7 14 21 27)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EXTRACTION_BATCH_SIZE,
        help="Batch size for extraction (reduce if OOM)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit number of pairs (for debugging)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/hidden_states/{pair_type}/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    # Determine pair type from filename for output subdir
    pair_type = args.pairs.stem.replace("_pairs", "")
    output_dir = args.output_dir or HIDDEN_DIR / pair_type

    logger.info("Pairs: %s", args.pairs)
    logger.info("Layers: %s", args.layers)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Output: %s", output_dir)

    # Lazy import to avoid loading torch/model until args are parsed
    from temporal_vectors.analysis.hidden_states import HiddenStateExtractor
    from temporal_vectors.models.loader import load_model_and_tokenizer

    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    extractor = HiddenStateExtractor(
        model=model,
        tokenizer=tokenizer,
        layers=args.layers,
    )

    logger.info("Starting extraction...")
    result = extractor.extract_dataset(
        pairs_path=args.pairs,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
    )

    # Summary
    logger.info("Extraction complete:")
    for layer_idx, tensor in result.items():
        logger.info("  Layer %d: %s", layer_idx, tensor.shape)

    # Sanity checks
    for layer_idx, tensor in result.items():
        if tensor.numel() > 0:
            if torch.isnan(tensor).any():
                logger.error("NaN detected in layer %d!", layer_idx)
            if torch.isinf(tensor).any():
                logger.error("Inf detected in layer %d!", layer_idx)

    logger.info("Done.")


if __name__ == "__main__":
    import torch

    main()
