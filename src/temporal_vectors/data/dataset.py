"""Torch Dataset and DataLoader helpers for counterfactual pairs."""

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from temporal_vectors.utils.io import read_jsonl


class PairDataset(Dataset):
    """Dataset that yields individual texts from counterfactual pairs.

    Each pair produces two items: the old text and the new text.
    Metadata is preserved so results can be mapped back to pairs.

    Args:
        pairs: List of pair dicts with 'text_old' and 'text_new' keys.
    """

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self.entries: list[dict[str, Any]] = []
        for pair in pairs:
            self.entries.append({
                "text": pair["text_old"],
                "pair_id": pair["id"],
                "side": "old",
                "domain": pair.get("domain", "unknown"),
                "year": pair.get("year_old"),
            })
            self.entries.append({
                "text": pair["text_new"],
                "pair_id": pair["id"],
                "side": "new",
                "domain": pair.get("domain", "unknown"),
                "year": pair.get("year_new"),
            })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.entries[idx]

    @classmethod
    def from_jsonl(cls, path: Path) -> "PairDataset":
        """Load a PairDataset from a JSONL file.

        Args:
            path: Path to the .jsonl pair file.

        Returns:
            PairDataset instance.
        """
        pairs = read_jsonl(path)
        return cls(pairs)


def text_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list]:
    """Collate function that groups dict fields into lists.

    Args:
        batch: List of entry dicts from PairDataset.

    Returns:
        Dict with lists for each key (texts, pair_ids, sides, etc.).
    """
    collated: dict[str, list] = {}
    for key in batch[0]:
        collated[key] = [entry[key] for entry in batch]
    return collated


def make_dataloader(
    dataset: PairDataset,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for text extraction.

    Args:
        dataset: A PairDataset instance.
        batch_size: Texts per batch.
        num_workers: DataLoader workers (0 = main process).

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=text_collate_fn,
    )
