"""Counterfactual pair construction and filtering.

Takes raw (old_sentence, new_sentence) tuples from the Wikipedia fetcher
and applies quality filters to produce valid JSONL pair records.
"""

import logging
import re
from pathlib import Path
from typing import Any

from temporal_vectors.utils.io import write_jsonl

logger = logging.getLogger(__name__)

# ── Filtering thresholds ─────────────────────────────────────────
MIN_TOKENS = 10
MAX_TOKENS = 200
MIN_CHANGE_RATIO = 0.05  # at least 5% of tokens must differ
MAX_CHANGE_RATIO = 0.90  # at most 90% differ (otherwise it's a rewrite, not an edit)


def _word_count(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(text.split())


def _change_ratio(old: str, new: str) -> float:
    """Fraction of tokens that differ between old and new."""
    old_tokens = old.lower().split()
    new_tokens = new.lower().split()
    if not old_tokens or not new_tokens:
        return 1.0
    common = set(old_tokens) & set(new_tokens)
    total = max(len(old_tokens), len(new_tokens))
    return 1.0 - len(common) / total


def _is_formatting_only(old: str, new: str) -> bool:
    """Check if the difference is purely formatting (punctuation, whitespace)."""
    normalise = lambda s: re.sub(r"[^a-zA-Z0-9]", "", s.lower())
    return normalise(old) == normalise(new)


def _classify_change(old: str, new: str) -> str:
    """Heuristic classification of what type of temporal change occurred.

    Returns:
        One of: "tense", "status", "numerical", "factual", "mixed".
    """
    tense_patterns = [
        (r"\bis\b", r"\bwas\b"),
        (r"\bwill\b", r"\bwould\b"),
        (r"\bwill be\b", r"\bis\b"),
        (r"\bis\b", r"\bwill be\b"),
        (r"\bare\b", r"\bwere\b"),
        (r"\bhas\b", r"\bhad\b"),
        (r"\bcurrent\b", r"\bformer\b"),
        (r"\bupcoming\b", r"\bprevious\b"),
    ]
    old_l, new_l = old.lower(), new.lower()
    for pat_old, pat_new in tense_patterns:
        if re.search(pat_old, old_l) and re.search(pat_new, new_l):
            return "tense"
        if re.search(pat_new, old_l) and re.search(pat_old, new_l):
            return "tense"

    # Check for status words
    status_words = ["current", "former", "incumbent", "elect", "acting", "designate"]
    if any(w in old_l or w in new_l for w in status_words):
        return "status"

    # Check for numerical changes (years, counts, percentages)
    old_nums = set(re.findall(r"\b\d+\.?\d*\b", old))
    new_nums = set(re.findall(r"\b\d+\.?\d*\b", new))
    if old_nums != new_nums:
        return "numerical"

    return "factual"


def filter_pair(raw_pair: dict[str, Any]) -> bool:
    """Apply quality filters to a single raw pair.

    Args:
        raw_pair: Dict with at least 'text_old' and 'text_new' keys.

    Returns:
        True if the pair passes all filters.
    """
    old = raw_pair["text_old"]
    new = raw_pair["text_new"]

    # Length filters
    if _word_count(old) < MIN_TOKENS or _word_count(new) < MIN_TOKENS:
        return False
    if _word_count(old) > MAX_TOKENS or _word_count(new) > MAX_TOKENS:
        return False

    # Must be a real content change, not just formatting
    if _is_formatting_only(old, new):
        return False

    # Change ratio: not too small (trivial) or too large (full rewrite)
    ratio = _change_ratio(old, new)
    if ratio < MIN_CHANGE_RATIO or ratio > MAX_CHANGE_RATIO:
        return False

    # Must not be identical
    if old.strip() == new.strip():
        return False

    return True


def build_pairs(
    raw_pairs: list[dict[str, Any]],
    pair_type: str = "natural",
    start_id: int = 1,
) -> list[dict[str, Any]]:
    """Filter raw pairs and assign IDs + metadata.

    Args:
        raw_pairs: List of dicts with keys: text_old, text_new, article,
            domain, year_old, year_new.
        pair_type: One of "natural", "synthetic", "control".
        start_id: Starting ID number for the pair sequence.

    Returns:
        List of valid pair records ready for JSONL output.
    """
    prefix = {"natural": "nat", "synthetic": "syn", "control": "ctl"}[pair_type]
    valid = []
    skipped = 0

    for raw in raw_pairs:
        if not filter_pair(raw):
            skipped += 1
            continue

        pair_id = f"{prefix}_{start_id + len(valid):04d}"
        change_type = _classify_change(raw["text_old"], raw["text_new"])

        record = {
            "id": pair_id,
            "type": pair_type,
            "domain": raw.get("domain", "unknown"),
            "text_old": raw["text_old"],
            "text_new": raw["text_new"],
            "year_old": raw.get("year_old"),
            "year_new": raw.get("year_new"),
            "article": raw.get("article", ""),
            "metadata": {
                "change_type": change_type,
                "revision_ids": raw.get("metadata", {}).get("revision_ids", []),
            },
        }
        valid.append(record)

    logger.info(
        "Built %d %s pairs (%d filtered out of %d raw)",
        len(valid),
        pair_type,
        skipped,
        len(raw_pairs),
    )
    return valid


def validate_pair_schema(record: dict[str, Any]) -> list[str]:
    """Validate that a pair record has all required fields with correct types.

    Args:
        record: A single pair dict.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    required_str = ["id", "type", "domain", "text_old", "text_new"]
    required_int = ["year_old", "year_new"]

    for field in required_str:
        if field not in record:
            errors.append(f"missing field: {field}")
        elif not isinstance(record[field], str):
            errors.append(f"{field} must be str, got {type(record[field]).__name__}")

    for field in required_int:
        if field not in record:
            errors.append(f"missing field: {field}")
        elif record[field] is not None and not isinstance(record[field], int):
            errors.append(f"{field} must be int, got {type(record[field]).__name__}")

    if record.get("type") not in ("natural", "synthetic", "control"):
        errors.append(f"type must be natural/synthetic/control, got {record.get('type')}")

    return errors


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate an entire dataset and return a summary report.

    Args:
        records: List of pair dicts.

    Returns:
        Dict with keys: total, valid, invalid, errors, domain_distribution.
    """
    invalid = []
    domain_counts: dict[str, int] = {}

    for i, rec in enumerate(records):
        errs = validate_pair_schema(rec)
        if errs:
            invalid.append({"index": i, "id": rec.get("id"), "errors": errs})
        domain = rec.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    total = len(records)
    domain_pcts = {d: round(c / total * 100, 1) for d, c in domain_counts.items()} if total else {}

    return {
        "total": total,
        "valid": total - len(invalid),
        "invalid": len(invalid),
        "errors": invalid[:20],  # first 20 errors for inspection
        "domain_distribution": domain_counts,
        "domain_percentages": domain_pcts,
    }


def save_pairs(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save pair records to a JSONL file.

    Args:
        records: List of validated pair dicts.
        output_path: Destination .jsonl file path.
    """
    write_jsonl(records, output_path)
    logger.info("Saved %d pairs to %s", len(records), output_path)
