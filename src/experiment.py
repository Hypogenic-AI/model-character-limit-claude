"""
Character Tracking Experiment
Tests how many characters LLMs can track in synthetic narratives.
"""

import json
import os
import time
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import httpx

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
@dataclass
class Config:
    """Experiment configuration."""
    dataset_path: str = "datasets/character_tracking_synthetic.json"
    results_dir: str = "results"
    temperature: float = 0.0
    max_tokens: int = 50
    timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 2.0

config = Config()

# Prompt template
PROMPT_TEMPLATE = """Read the following story and answer the question based on the final state of affairs.

Story:
{story}

Question: {question}

Answer with ONLY the answer word (e.g., "kitchen", "happy", "book", or "nothing"). Do not include any explanation."""


def load_dataset(path: str) -> dict:
    """Load the synthetic character tracking dataset."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data['examples'])} examples")
    return data


class OpenAIClient:
    """OpenAI API client with retry logic."""

    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def query(self, story: str, question: str) -> str:
        """Query the model with a story and question."""
        prompt = PROMPT_TEMPLATE.format(story=story, question=question)

        for attempt in range(config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return response.choices[0].message.content.strip().lower()
            except Exception as e:
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
                else:
                    print(f"Error after {config.retry_attempts} attempts: {e}")
                    return "[ERROR]"
        return "[ERROR]"


class OpenRouterClient:
    """OpenRouter API client for Claude models."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def query(self, story: str, question: str) -> str:
        """Query the model with a story and question."""
        prompt = PROMPT_TEMPLATE.format(story=story, question=question)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        for attempt in range(config.retry_attempts):
            try:
                with httpx.Client(timeout=config.timeout) as client:
                    response = client.post(self.base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip().lower()
            except Exception as e:
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
                else:
                    print(f"Error after {config.retry_attempts} attempts: {e}")
                    return "[ERROR]"
        return "[ERROR]"


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.lower().strip()
    # Remove punctuation and extra whitespace
    answer = answer.replace(".", "").replace(",", "").strip()
    # Handle common variations
    if answer in ["none", "nothing", "no object", "no item", "null"]:
        return "nothing"
    # Extract single word answers if model gives explanation
    words = answer.split()
    if len(words) > 0:
        # Return first word for simple answers
        return words[0]
    return answer


def evaluate_answer(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    return pred_norm == exp_norm


def run_experiment(client, data: dict, model_name: str) -> pd.DataFrame:
    """Run the experiment for a given model."""
    results = []

    print(f"\nRunning experiment with {model_name}...")

    for example in tqdm(data["examples"], desc=model_name):
        story = example["story"]
        num_characters = example["num_characters"]
        num_actions = example["num_actions"]

        for q in example["questions"]:
            question = q["question"]
            expected = q["answer"]
            q_type = q["type"]

            # Query the model
            predicted = client.query(story, question)
            correct = evaluate_answer(predicted, expected)

            results.append({
                "model": model_name,
                "num_characters": num_characters,
                "num_actions": num_actions,
                "question_type": q_type,
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "story_len": len(story),
            })

            # Small delay to avoid rate limits
            time.sleep(0.1)

    return pd.DataFrame(results)


def compute_baselines(data: dict) -> pd.DataFrame:
    """Compute baseline accuracy metrics."""
    results = []

    # Collect all valid answers for random baseline
    all_locations = set()
    all_moods = set()
    all_holdings = set()

    for example in data["examples"]:
        for char, state in example["final_states"].items():
            all_locations.add(state["location"])
            all_moods.add(state["mood"])
            if state["holding"]:
                all_holdings.add(state["holding"])
    all_holdings.add("nothing")

    all_locations = list(all_locations)
    all_moods = list(all_moods)
    all_holdings = list(all_holdings)

    for example in data["examples"]:
        num_characters = example["num_characters"]
        num_actions = example["num_actions"]

        # Parse initial states from first sentences
        initial_states = {}
        for char in example["characters"]:
            # Find first sentence about this character
            for sent in example["sentences"]:
                if sent.startswith(char):
                    # Parse initial location and mood
                    # Format: "Alice is in the kitchen and feels happy."
                    parts = sent.split(" is in the ")
                    if len(parts) == 2:
                        loc_mood = parts[1]
                        loc = loc_mood.split(" and feels ")[0]
                        mood = loc_mood.split(" and feels ")[1].replace(".", "")
                        initial_states[char] = {
                            "location": loc,
                            "mood": mood,
                            "holding": "nothing"
                        }
                    break

        for q in example["questions"]:
            question = q["question"]
            expected = q["answer"]
            q_type = q["type"]

            # Extract character from question
            char = question.replace("Where is ", "").replace("How does ", "").replace(" feel?", "").replace("What is ", "").replace(" holding?", "").replace("?", "")

            # Random baseline
            if q_type == "location":
                random_pred = random.choice(all_locations)
            elif q_type == "mood":
                random_pred = random.choice(all_moods)
            else:
                random_pred = random.choice(all_holdings)

            # First-state baseline
            first_state_pred = "unknown"
            if char in initial_states:
                if q_type == "location":
                    first_state_pred = initial_states[char]["location"]
                elif q_type == "mood":
                    first_state_pred = initial_states[char]["mood"]
                else:
                    first_state_pred = initial_states[char]["holding"]

            results.append({
                "model": "random_baseline",
                "num_characters": num_characters,
                "num_actions": num_actions,
                "question_type": q_type,
                "expected": expected,
                "predicted": random_pred,
                "correct": random_pred == expected,
            })

            results.append({
                "model": "first_state_baseline",
                "num_characters": num_characters,
                "num_actions": num_actions,
                "question_type": q_type,
                "expected": expected,
                "predicted": first_state_pred,
                "correct": first_state_pred == expected,
            })

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> dict:
    """Analyze experiment results."""
    analysis = {}

    # Overall accuracy by model
    analysis["overall_accuracy"] = df.groupby("model")["correct"].mean().to_dict()

    # Accuracy by character count and model
    char_acc = df.groupby(["model", "num_characters"])["correct"].mean().unstack(level=0)
    analysis["accuracy_by_characters"] = char_acc.to_dict()

    # Accuracy by question type and model
    type_acc = df.groupby(["model", "question_type"])["correct"].mean().unstack(level=0)
    analysis["accuracy_by_question_type"] = type_acc.to_dict()

    # Accuracy by action count and model
    action_acc = df.groupby(["model", "num_actions"])["correct"].mean().unstack(level=0)
    analysis["accuracy_by_actions"] = action_acc.to_dict()

    return analysis


def save_results(df: pd.DataFrame, analysis: dict, results_dir: str = "results"):
    """Save results to files."""
    os.makedirs(results_dir, exist_ok=True)

    # Save raw results
    df.to_csv(f"{results_dir}/raw_results.csv", index=False)

    # Save analysis
    with open(f"{results_dir}/analysis.json", "w") as f:
        # Convert any numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(analysis), f, indent=2)

    print(f"\nResults saved to {results_dir}/")


def main():
    """Main experiment runner."""
    print("="*60)
    print("Character Tracking Experiment")
    print("="*60)

    # Load data
    data = load_dataset(config.dataset_path)

    # Initialize clients
    models = {
        "gpt-4.1": OpenAIClient("gpt-4.1"),
        "gpt-3.5-turbo": OpenAIClient("gpt-3.5-turbo"),
        # "claude-sonnet-4": OpenRouterClient("anthropic/claude-sonnet-4"),
    }

    # Run experiments
    all_results = []

    for model_name, client in models.items():
        try:
            results_df = run_experiment(client, data, model_name)
            all_results.append(results_df)
        except Exception as e:
            print(f"Error running {model_name}: {e}")

    # Add baselines
    print("\nComputing baselines...")
    baseline_df = compute_baselines(data)
    all_results.append(baseline_df)

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Analyze
    print("\nAnalyzing results...")
    analysis = analyze_results(combined_df)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nOverall Accuracy by Model:")
    for model, acc in sorted(analysis["overall_accuracy"].items()):
        print(f"  {model}: {acc:.1%}")

    # Save
    save_results(combined_df, analysis)

    return combined_df, analysis


if __name__ == "__main__":
    main()
