#!/usr/bin/env python
"""Temporal walk: sweep alpha to see if events unfold chronologically.

Use neutral prompts with NO date anchor so the temporal vector alone
controls the temporal frame. Sweep from negative (push backward) to
positive (push forward) alpha.
"""

import json
import logging
from pathlib import Path

import torch

from temporal_vectors.config import VECTOR_DIR, RESULTS_DIR, SEED
from temporal_vectors.utils.reproducibility import seed_everything
from temporal_vectors.models.loader import load_model_and_tokenizer
from temporal_vectors.models.hooks import steer_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

#soft temporal anchors — elicit specific names without hard-locking a year
#negative alpha = push backward in time, positive = push forward
WALK_PROMPTS = [
    {
        "prompt": "The current president of the United States is",
        "expected_sequence": ["Bush", "Obama", "Trump", "Biden"],
        "domain": "politics",
    },
    {
        "prompt": "Today, the prime minister of the United Kingdom is",
        "expected_sequence": ["Blair", "Brown", "Cameron", "May", "Johnson", "Starmer"],
        "domain": "politics",
    },
    {
        "prompt": "The latest iPhone model is the iPhone",
        "expected_sequence": ["5", "6", "7", "8", "X", "11", "12", "13", "14", "15", "16"],
        "domain": "technology",
    },
    {
        "prompt": "Right now, the chancellor of Germany is",
        "expected_sequence": ["Merkel", "Scholz"],
        "domain": "politics",
    },
    {
        "prompt": "The current richest person in the world is",
        "expected_sequence": ["Gates", "Bezos", "Musk"],
        "domain": "economics",
    },
    {
        "prompt": "The most recent Summer Olympics were held in",
        "expected_sequence": ["Beijing", "London", "Rio", "Tokyo", "Paris"],
        "domain": "sports",
    },
    {
        "prompt": "The current CEO of Twitter is",
        "expected_sequence": ["Dorsey", "Agrawal", "Musk"],
        "domain": "technology",
    },
]

#sweep negative (backward) through positive (forward)
ALPHAS = [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def generate(model, tokenizer, prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def check_sequence(text, expected_sequence):
    """check which expected items appear in the generated text."""
    found = []
    for item in expected_sequence:
        if item.lower() in text.lower():
            found.append(item)
    return found


def main():
    seed_everything(SEED)

    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    layer = 14
    direction = torch.load(
        VECTOR_DIR / "synthetic" / f"temporal_direction_layer_{layer}.pt",
        map_location="cpu", weights_only=False,
    ).to(model.device)

    all_results = []

    for walk in WALK_PROMPTS:
        prompt = walk["prompt"]
        expected = walk["expected_sequence"]

        print(f"\n{'=' * 90}")
        print(f"TEMPORAL WALK: {prompt}")
        print(f"Expected chronological sequence: {' -> '.join(expected)}")
        print(f"{'=' * 90}")

        walk_result = {
            "prompt": prompt,
            "expected_sequence": expected,
            "domain": walk["domain"],
            "outputs": {},
        }

        for alpha in ALPHAS:
            if alpha == 0.0:
                text = generate(model, tokenizer, prompt)
            else:
                with steer_model(model, layer, direction, alpha=alpha, token_position=None):
                    text = generate(model, tokenizer, prompt)

            #extract just the generated part (after the prompt)
            generated = text[len(prompt):].strip()
            found = check_sequence(text, expected)

            print(f"  alpha={alpha:>5.1f}: {generated[:80]:<80s} | found: {found}")
            walk_result["outputs"][str(alpha)] = {
                "full_text": text,
                "generated": generated,
                "found_items": found,
            }

        all_results.append(walk_result)

    #save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "temporal_walk_outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
