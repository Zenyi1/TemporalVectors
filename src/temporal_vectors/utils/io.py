"""I/O utilities: JSONL read/write and checkpoint helpers."""

import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects, one per line.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of JSON-serialisable dicts.
        path: Output file path (parent dirs created automatically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(
    tensor: torch.Tensor,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a tensor checkpoint with optional metadata.

    Args:
        tensor: The tensor to save.
        path: Output .pt file path.
        metadata: Optional dict saved alongside the tensor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tensor": tensor}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a tensor checkpoint saved by save_checkpoint.

    Args:
        path: Path to the .pt file.

    Returns:
        Dict with keys 'tensor' and optionally 'metadata'.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def save_results(results: dict[str, Any], path: Path) -> None:
    """Save a results dict as formatted JSON.

    Args:
        results: Results dictionary (must be JSON-serialisable).
        path: Output .json file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def load_results(path: Path) -> dict[str, Any]:
    """Load a results JSON file.

    Args:
        path: Path to the .json file.

    Returns:
        Parsed results dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
