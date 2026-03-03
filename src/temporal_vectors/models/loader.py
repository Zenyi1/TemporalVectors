"""Model and tokenizer loading utilities."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from temporal_vectors.config import DATA_CACHE, MODEL_NAME

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str | None = None,
    device_map: str = "auto",
    cache_dir: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer in eval mode with no grad.

    The model is loaded in bfloat16 by default for memory efficiency.
    Hidden-state extraction should cast to float32 downstream for
    numerical precision (see analysis/hidden_states.py).

    Args:
        model_name: HuggingFace model identifier. Defaults to config.MODEL_NAME.
        device_map: Device placement strategy ("auto", "cpu", "cuda:0", etc.).
        cache_dir: Directory for cached model weights. Defaults to config.DATA_CACHE.
        torch_dtype: Weight dtype. Use bfloat16 to save VRAM; extraction
            code is responsible for casting hidden states to float32.

    Returns:
        Tuple of (model, tokenizer), model in eval mode.
    """
    model_name = model_name or MODEL_NAME
    cache_dir = cache_dir or str(DATA_CACHE)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, torch_dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.1fB parameters", param_count / 1e9)

    return model, tokenizer
