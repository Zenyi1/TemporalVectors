#!/usr/bin/env python
"""Demo: steer LLaMA generation with the temporal vector.

Feed the model a prompt about an old fact, generate with and without
the temporal direction, compare outputs.
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

PROMPTS = [
    "As of 2020, the president of the United States is",
    "As of 2019, the prime minister of the United Kingdom is",
    "As of 2020, the latest iPhone model is the iPhone",
    "The 2020 Olympics were held in",
    "As of 2019, the World Health Organization has declared",
    "In 2020, the most popular social media platform owned by ByteDance is",
]


def generate(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    seed_everything(SEED)

    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    #try multiple layers and alphas
    layers = [14, 21, 27]
    alphas = [1.0, 2.0, 5.0, 10.0]
    all_outputs = []

    for layer in layers:
        direction = torch.load(
            VECTOR_DIR / "synthetic" / f"temporal_direction_layer_{layer}.pt",
            map_location="cpu", weights_only=False,
        ).to(model.device)

        for prompt in PROMPTS:
            print(f"\n{'=' * 80}")
            print(f"PROMPT: {prompt}")
            print(f"LAYER: {layer}")
            print(f"{'=' * 80}")

            #baseline (no steering)
            baseline = generate(model, tokenizer, prompt)
            print(f"\n  No steering:")
            print(f"    {baseline}")

            entry = {"prompt": prompt, "layer": layer, "baseline": baseline, "steered": {}}

            #steered at different alphas
            for alpha in alphas:
                with steer_model(model, layer, direction, alpha=alpha, token_position=None):
                    steered = generate(model, tokenizer, prompt)
                print(f"\n  alpha={alpha}:")
                print(f"    {steered}")
                entry["steered"][str(alpha)] = steered

            all_outputs.append(entry)

        print("\n")

    #save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "steering_demo_outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
