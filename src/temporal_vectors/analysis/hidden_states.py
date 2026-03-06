"""Hidden state extraction from transformer layers.

Extracts per-layer hidden states for batches of texts, with last-token
pooling, checkpointing, and memory management.
"""

import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from temporal_vectors.config import EXTRACTION_BATCH_SIZE, MAX_SEQ_LENGTH, TARGET_LAYERS
from temporal_vectors.data.dataset import PairDataset, make_dataloader
from temporal_vectors.models.hooks import capture_hidden_states

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """Extract hidden states from specified transformer layers.

    Uses forward hooks to capture hidden states during inference,
    with last-token pooling (standard for causal LMs). All hidden
    states are stored in float32 for numerical precision.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: The corresponding tokenizer.
        layers: Which layers to extract from. Defaults to config.TARGET_LAYERS.
        pooling: Pooling strategy. Only "last_token" is supported.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layers: list[int] | None = None,
        pooling: str = "last_token",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or TARGET_LAYERS
        self.pooling = pooling

    def extract(self, text: str) -> dict[int, torch.Tensor]:
        """Extract hidden states for a single text.

        Args:
            text: Input text string.

        Returns:
            Dict mapping layer index to pooled hidden state, shape (hidden_size,).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        ).to(self.model.device)

        with torch.no_grad(), capture_hidden_states(self.model, self.layers) as captured:
            self.model(**inputs)

        result = {}
        for layer_idx, hidden in captured.items():
            # hidden shape: (1, seq_len, hidden_size) -- already float32 from hook
            if self.pooling == "last_token":
                seq_len = inputs["attention_mask"].sum(dim=1) - 1  # last real token
                result[layer_idx] = hidden[0, seq_len.item(), :]  # (hidden_size,)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return result

    def extract_batch(self, texts: list[str]) -> dict[int, torch.Tensor]:
        """Extract hidden states for a batch of texts.

        Args:
            texts: List of input text strings.

        Returns:
            Dict mapping layer index to tensor of shape (batch_size, hidden_size).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
        ).to(self.model.device)

        with torch.no_grad(), capture_hidden_states(self.model, self.layers) as captured:
            self.model(**inputs)

        result = {}
        for layer_idx, hidden in captured.items():
            # hidden shape: (batch, seq_len, hidden_size)
            if self.pooling == "last_token":
                # Find last non-padding token for each sequence
                seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # (batch,)
                batch_indices = torch.arange(hidden.size(0))
                result[layer_idx] = hidden[batch_indices, seq_lengths.cpu(), :]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return result

    def extract_dataset(
        self,
        pairs_path: Path,
        output_dir: Path,
        batch_size: int = EXTRACTION_BATCH_SIZE,
        max_pairs: int | None = None,
        checkpoint_every: int = 100,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states for an entire pair dataset.

        Processes both old and new texts, saves per-layer .pt files.
        Supports checkpointing and resumption.

        Args:
            pairs_path: Path to the JSONL pair file.
            output_dir: Directory to save layer_{L}.pt files.
            batch_size: Number of texts per batch.
            max_pairs: Limit number of pairs (for debugging).
            checkpoint_every: Save checkpoint every N batches.

        Returns:
            Dict mapping layer index to tensor of shape (N_texts, hidden_size).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = PairDataset.from_jsonl(pairs_path)
        if max_pairs:
            dataset.entries = dataset.entries[: max_pairs * 2]

        dataloader = make_dataloader(dataset, batch_size=batch_size)

        # Check for existing checkpoint
        checkpoint_path = output_dir / "_checkpoint.pt"
        start_batch = 0
        all_states: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}

        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            start_batch = ckpt["batch_idx"] + 1
            all_states = ckpt["states"]
            logger.info("Resuming from batch %d", start_batch)

        total_batches = len(dataloader)
        logger.info(
            "Extracting %d texts in %d batches (layers=%s)",
            len(dataset),
            total_batches,
            self.layers,
        )

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            if batch_idx < start_batch:
                continue

            texts = batch["text"]
            states = self.extract_batch(texts)

            for layer_idx, tensor in states.items():
                all_states[layer_idx].append(tensor)

            # Checkpoint periodically
            if (batch_idx + 1) % checkpoint_every == 0:
                torch.save(
                    {"batch_idx": batch_idx, "states": all_states},
                    checkpoint_path,
                )
                logger.info("Checkpoint saved at batch %d/%d", batch_idx + 1, total_batches)

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batches per layer
        result = {}
        for layer_idx in self.layers:
            if all_states[layer_idx]:
                result[layer_idx] = torch.cat(all_states[layer_idx], dim=0)
            else:
                result[layer_idx] = torch.empty(0)

        # Save per-layer files
        metadata = {
            "source": str(pairs_path),
            "n_texts": len(dataset),
            "layers": self.layers,
            "pooling": self.pooling,
        }
        for layer_idx, tensor in result.items():
            layer_path = output_dir / f"layer_{layer_idx}.pt"
            torch.save({"tensor": tensor, "metadata": metadata}, layer_path)
            logger.info("Saved layer %d: shape %s -> %s", layer_idx, tensor.shape, layer_path)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return result
