"""
Extended experiment: push to higher character counts (48, 64) and
also test with more state changes per character to stress-test tracking.
"""

import os
import json
import sys
import random
import numpy as np
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from story_generator import generate_dataset, StoryInstance
from experiment import build_prompt, call_api, parse_response, evaluate_story, normalize, compute_baselines

random.seed(42)
np.random.seed(42)

RESULTS_DIR = "/workspaces/model-character-limit-claude/results"


def run_extended():
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_key)

    # Extended character counts
    char_counts = [2, 4, 8, 16, 32, 48, 64]
    stories_per = 3  # fewer stories for speed at high counts

    models = {
        "gpt-4.1-mini": {"client": openai_client, "model_id": "gpt-4.1-mini"},
        "gpt-4.1": {"client": openai_client, "model_id": "gpt-4.1"},
    }

    print("Generating extended dataset...")
    dataset = generate_dataset(
        character_counts=char_counts,
        stories_per_count=stories_per,
        state_changes_per_char=2,
    )
    print(f"Generated {len(dataset)} stories ({sum(len(s.questions) for s in dataset)} questions)")

    all_results = {}

    for model_name, cfg in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        results = []

        for story in tqdm(dataset, desc=model_name):
            sys_msg, usr_msg = build_prompt(story)
            # Increase max_tokens for larger character sets
            max_tok = min(8192, 256 + story.num_characters * 100)
            response = call_api(cfg["client"], cfg["model_id"], sys_msg, usr_msg, max_tokens=max_tok)

            char_names = [cs["name"] for cs in story.characters]
            model_answers = parse_response(response, char_names)

            story_results = evaluate_story(story, model_answers)
            for r in story_results:
                r["model"] = model_name
            results.extend(story_results)

        all_results[model_name] = results

        # Print summary
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        print(f"  Overall: {correct}/{total} = {correct/total:.1%}")
        for nc in char_counts:
            nc_res = [r for r in results if r["num_characters"] == nc]
            nc_ok = sum(1 for r in nc_res if r["is_correct"])
            if nc_res:
                print(f"  {nc:2d} chars: {nc_ok}/{len(nc_res)} = {nc_ok/len(nc_res):.1%}")

    # Save extended results
    combined = []
    for results in all_results.values():
        combined.extend(results)
    with open(f"{RESULTS_DIR}/extended_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Baselines
    baselines = compute_baselines(dataset)
    with open(f"{RESULTS_DIR}/extended_baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nBaselines - Random: {baselines['random']:.1%}, Initial: {baselines['initial_state']:.1%}")

    # Also test with MORE state changes (stress test)
    print("\n\n" + "="*60)
    print("STRESS TEST: 4 state changes per character")
    print("="*60)

    stress_dataset = generate_dataset(
        character_counts=[4, 8, 16, 32],
        stories_per_count=3,
        state_changes_per_char=4,
        base_seed=99,
    )
    print(f"Generated {len(stress_dataset)} stress-test stories")

    stress_results = {}
    for model_name, cfg in models.items():
        print(f"\nModel: {model_name}")
        results = []
        for story in tqdm(stress_dataset, desc=f"{model_name}-stress"):
            sys_msg, usr_msg = build_prompt(story)
            max_tok = min(8192, 256 + story.num_characters * 100)
            response = call_api(cfg["client"], cfg["model_id"], sys_msg, usr_msg, max_tokens=max_tok)
            char_names = [cs["name"] for cs in story.characters]
            model_answers = parse_response(response, char_names)
            story_results = evaluate_story(story, model_answers)
            for r in story_results:
                r["model"] = model_name
                r["state_changes"] = 4
            results.extend(story_results)

        stress_results[model_name] = results
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        print(f"  Overall: {correct}/{total} = {correct/total:.1%}")
        for nc in [4, 8, 16, 32]:
            nc_res = [r for r in results if r["num_characters"] == nc]
            nc_ok = sum(1 for r in nc_res if r["is_correct"])
            if nc_res:
                print(f"  {nc:2d} chars: {nc_ok}/{len(nc_res)} = {nc_ok/len(nc_res):.1%}")

    # Save stress test
    stress_combined = []
    for results in stress_results.values():
        stress_combined.extend(results)
    with open(f"{RESULTS_DIR}/stress_results.json", "w") as f:
        json.dump(stress_combined, f, indent=2)


if __name__ == "__main__":
    run_extended()
