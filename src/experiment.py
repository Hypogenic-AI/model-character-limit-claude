"""
Character Tracking Experiment
Tests how many characters frontier LLMs can track in synthetic narratives.
Uses batch JSON queries for efficiency and detailed error analysis.
"""

import os
import json
import time
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Ensure we can import from src/
sys.path.insert(0, os.path.dirname(__file__))
from story_generator import generate_dataset, StoryInstance

# Reproducibility
random.seed(42)
np.random.seed(42)

RESULTS_DIR = "/workspaces/model-character-limit-claude/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)


def build_prompt(story: StoryInstance) -> tuple[str, str]:
    """Build prompt asking about ALL characters at once via JSON."""
    system_msg = (
        "You are a careful reader. Read the story and answer questions about "
        "each character's current state at the END of the story. "
        "Respond ONLY with a JSON object, no explanation."
    )

    # Build expected JSON template
    template = {}
    for cs in story.characters:
        template[cs["name"]] = {"location": "...", "possession": "..."}

    user_msg = f"""Read the following story carefully, then answer what each character's final location and possession is.

STORY:
{story.story_text}

Answer in this exact JSON format (replace "..." with the actual answers):
{json.dumps(template, indent=2)}

Important: Track all state changes through the story. Give the FINAL state for each character.
Respond with ONLY the JSON object."""

    return system_msg, user_msg


def call_api(client: OpenAI, model: str, system_msg: str, user_msg: str,
             temperature: float = 0, max_tokens: int = 4096) -> str:
    """Call OpenAI-compatible API with retry."""
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  API error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after 5 attempts: {e}")
                return ""


def parse_response(response: str, character_names: list[str]) -> dict:
    """Parse JSON response into character -> {location, possession} dict."""
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    result = {}
    for name in character_names:
        if name in parsed and isinstance(parsed[name], dict):
            result[name] = {
                "location": str(parsed[name].get("location", "")),
                "possession": str(parsed[name].get("possession", "")),
            }
    return result


def normalize(answer: str) -> str:
    """Normalize answer for comparison."""
    a = answer.lower().strip().rstrip(".")
    for prefix in ["a ", "an ", "the "]:
        if a.startswith(prefix):
            a = a[len(prefix):]
    return a.strip()


def evaluate_story(story: StoryInstance, model_answers: dict) -> list[dict]:
    """Evaluate model answers against ground truth with error analysis."""
    results = []
    for q, gt in zip(story.questions, story.ground_truth):
        char_name = q["character"]
        attribute = q["attribute"]
        correct_answer = gt["answer"]

        model_ans = ""
        if char_name in model_answers:
            model_ans = model_answers[char_name].get(attribute, "")

        is_correct = normalize(model_ans) == normalize(correct_answer)

        # Error analysis
        error_type = None
        confused_with = None
        if not is_correct and model_ans:
            # Check forgetting (reverted to initial state)
            for init in story.initial_states:
                if init["name"] == char_name:
                    if normalize(model_ans) == normalize(init.get(attribute, "")):
                        error_type = "forgetting"
                    break

            # Check confusion (answer belongs to another character)
            if error_type is None:
                for cs in story.characters:
                    if cs["name"] != char_name:
                        if normalize(model_ans) == normalize(cs.get(attribute, "")):
                            error_type = "confusion"
                            confused_with = cs["name"]
                            break
                # Also check initial states of other characters
                if error_type is None:
                    for init in story.initial_states:
                        if init["name"] != char_name:
                            if normalize(model_ans) == normalize(init.get(attribute, "")):
                                error_type = "confusion"
                                confused_with = init["name"]
                                break

            if error_type is None:
                error_type = "other"
        elif not is_correct and not model_ans:
            error_type = "missing"

        # Get intro order
        intro_order = None
        for cs in story.characters:
            if cs["name"] == char_name:
                intro_order = cs["introduction_order"]
                break

        results.append({
            "character": char_name,
            "attribute": attribute,
            "correct_answer": correct_answer,
            "model_answer": model_ans,
            "is_correct": is_correct,
            "error_type": error_type,
            "confused_with": confused_with,
            "introduction_order": intro_order,
            "num_characters": story.num_characters,
        })

    return results


def run_experiment(
    models: dict,
    character_counts: list[int] = [2, 4, 8, 16, 32],
    stories_per_count: int = 5,
    state_changes: int = 2,
):
    """Run the full experiment."""
    print("Generating dataset...")
    dataset = generate_dataset(
        character_counts=character_counts,
        stories_per_count=stories_per_count,
        state_changes_per_char=state_changes,
    )
    print(f"Generated {len(dataset)} stories ({sum(len(s.questions) for s in dataset)} total questions)")

    # Save dataset for reproducibility
    dataset_records = []
    for story in dataset:
        dataset_records.append({
            "num_characters": story.num_characters,
            "story_text": story.story_text,
            "characters": story.characters,
            "initial_states": story.initial_states,
            "questions": story.questions,
            "ground_truth": story.ground_truth,
            "events": story.events,
        })
    with open(f"{RESULTS_DIR}/dataset.json", "w") as f:
        json.dump(dataset_records, f, indent=2)

    all_results = {}

    for model_name, model_config in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        client = model_config["client"]
        model_id = model_config["model_id"]
        results = []

        for story in tqdm(dataset, desc=f"{model_name}"):
            system_msg, user_msg = build_prompt(story)
            response = call_api(client, model_id, system_msg, user_msg)

            char_names = [cs["name"] for cs in story.characters]
            model_answers = parse_response(response, char_names)

            story_results = evaluate_story(story, model_answers)
            for r in story_results:
                r["model"] = model_name
                r["raw_response_preview"] = response[:300]
            results.extend(story_results)

        all_results[model_name] = results
        with open(f"{RESULTS_DIR}/results_{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        print(f"  Overall: {correct}/{total} = {correct/total:.1%}")
        for nc in character_counts:
            nc_res = [r for r in results if r["num_characters"] == nc]
            nc_ok = sum(1 for r in nc_res if r["is_correct"])
            print(f"  {nc:2d} chars: {nc_ok}/{len(nc_res)} = {nc_ok/len(nc_res):.1%}")

    # Save combined
    combined = []
    for results in all_results.values():
        combined.extend(results)
    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Compute baselines
    baselines = compute_baselines(dataset)
    with open(f"{RESULTS_DIR}/baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nBaselines - Random: {baselines['random']:.1%}, Initial-state: {baselines['initial_state']:.1%}")

    # Save config
    config = {
        "character_counts": character_counts,
        "stories_per_count": stories_per_count,
        "state_changes_per_char": state_changes,
        "models": list(models.keys()),
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
    }
    with open(f"{RESULTS_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return all_results, dataset


def compute_baselines(dataset: list[StoryInstance]) -> dict:
    """Compute random and initial-state baseline accuracy."""
    rng = random.Random(42)
    random_correct = 0
    initial_correct = 0
    total = 0

    for story in dataset:
        all_locs = list(set(cs["location"] for cs in story.characters))
        all_items = list(set(cs["possession"] for cs in story.characters))

        for q, gt in zip(story.questions, story.ground_truth):
            total += 1
            attr = q["attribute"]

            # Random baseline
            if attr == "location":
                if rng.choice(all_locs) == gt["answer"]:
                    random_correct += 1
            else:
                if rng.choice(all_items) == gt["answer"]:
                    random_correct += 1

            # Initial state baseline
            for init in story.initial_states:
                if init["name"] == q["character"]:
                    if normalize(init.get(attr, "")) == normalize(gt["answer"]):
                        initial_correct += 1
                    break

    return {
        "random": random_correct / total if total > 0 else 0,
        "initial_state": initial_correct / total if total > 0 else 0,
        "total_questions": total,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Character Tracking Limit Experiment")
    print("=" * 60)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)

    models = {
        "gpt-4.1-mini": {
            "client": openai_client,
            "model_id": "gpt-4.1-mini",
        },
        "gpt-4.1": {
            "client": openai_client,
            "model_id": "gpt-4.1",
        },
    }

    all_results, dataset = run_experiment(
        models=models,
        character_counts=[2, 4, 8, 16, 32],
        stories_per_count=5,
        state_changes=2,
    )

    print("\nExperiment complete! Results saved to results/")
