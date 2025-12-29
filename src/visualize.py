"""
Visualization script for character tracking experiment results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_results(results_dir: str = "results") -> tuple:
    """Load experiment results."""
    df = pd.read_csv(f"{results_dir}/raw_results.csv")
    with open(f"{results_dir}/analysis.json") as f:
        analysis = json.load(f)
    return df, analysis


def plot_accuracy_vs_characters(df: pd.DataFrame, save_dir: str = "results/figures"):
    """Plot accuracy vs number of characters for each model."""
    os.makedirs(save_dir, exist_ok=True)

    # Filter to just LLM models (not baselines)
    llm_models = [m for m in df["model"].unique() if "baseline" not in m]
    baseline_models = [m for m in df["model"].unique() if "baseline" in m]

    # Compute accuracy by character count
    acc_by_char = df.groupby(["model", "num_characters"])["correct"].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"gpt-4.1": "#1f77b4", "gpt-3.5-turbo": "#ff7f0e", "claude-sonnet-4": "#2ca02c",
              "random_baseline": "#d62728", "first_state_baseline": "#9467bd"}
    markers = {"gpt-4.1": "o", "gpt-3.5-turbo": "s", "claude-sonnet-4": "^",
               "random_baseline": "x", "first_state_baseline": "+"}

    for model in llm_models:
        model_data = acc_by_char[acc_by_char["model"] == model]
        ax.plot(model_data["num_characters"], model_data["mean"],
                marker=markers.get(model, "o"), label=model,
                color=colors.get(model, None), linewidth=2, markersize=8)
        # Add error bands (std error)
        stderr = model_data["std"] / np.sqrt(model_data["count"])
        ax.fill_between(model_data["num_characters"],
                       model_data["mean"] - stderr,
                       model_data["mean"] + stderr,
                       alpha=0.2, color=colors.get(model, None))

    # Add baseline reference lines
    for model in baseline_models:
        model_data = acc_by_char[acc_by_char["model"] == model]
        ax.plot(model_data["num_characters"], model_data["mean"],
                marker=markers.get(model, "."), label=model,
                color=colors.get(model, None), linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Number of Characters", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Character Tracking Accuracy vs. Number of Characters", fontsize=14)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 21)
    ax.grid(True, alpha=0.3)

    # Add horizontal line at chance level
    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='~Chance')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_vs_characters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/accuracy_vs_characters.png")


def plot_accuracy_by_question_type(df: pd.DataFrame, save_dir: str = "results/figures"):
    """Plot accuracy breakdown by question type."""
    os.makedirs(save_dir, exist_ok=True)

    llm_models = [m for m in df["model"].unique() if "baseline" not in m]

    # Compute accuracy by question type and model
    acc_by_type = df[df["model"].isin(llm_models)].groupby(["model", "question_type"])["correct"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(llm_models))
    width = 0.25

    question_types = ["location", "mood", "holding"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, q_type in enumerate(question_types):
        if q_type in acc_by_type.columns:
            values = [acc_by_type.loc[m, q_type] if m in acc_by_type.index else 0 for m in llm_models]
            ax.bar(x + i*width, values, width, label=q_type.capitalize(), color=colors[i])

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Question Type", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(llm_models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_by_question_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/accuracy_by_question_type.png")


def plot_accuracy_by_actions(df: pd.DataFrame, save_dir: str = "results/figures"):
    """Plot accuracy vs number of actions for different character counts."""
    os.makedirs(save_dir, exist_ok=True)

    llm_models = [m for m in df["model"].unique() if "baseline" not in m]

    # Focus on one model for cleaner visualization
    if "gpt-4.1" in llm_models:
        focus_model = "gpt-4.1"
    else:
        focus_model = llm_models[0]

    model_df = df[df["model"] == focus_model]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by character count
    char_groups = [2, 5, 10, 15, 20]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(char_groups)))

    for i, n_chars in enumerate(char_groups):
        if n_chars in model_df["num_characters"].values:
            subset = model_df[model_df["num_characters"] == n_chars]
            acc_by_action = subset.groupby("num_actions")["correct"].mean()
            ax.plot(acc_by_action.index, acc_by_action.values,
                   marker="o", label=f"{n_chars} characters",
                   color=colors[i], linewidth=2, markersize=8)

    ax.set_xlabel("Number of Actions", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy vs. Actions at Different Character Counts ({focus_model})", fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_by_actions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/accuracy_by_actions.png")


def plot_heatmap(df: pd.DataFrame, save_dir: str = "results/figures"):
    """Create heatmap of accuracy by characters and actions."""
    os.makedirs(save_dir, exist_ok=True)

    llm_models = [m for m in df["model"].unique() if "baseline" not in m]

    for model in llm_models:
        model_df = df[df["model"] == model]
        pivot = model_df.groupby(["num_characters", "num_actions"])["correct"].mean().unstack()

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy", fontsize=12)

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("Number of Actions", fontsize=12)
        ax.set_ylabel("Number of Characters", fontsize=12)
        ax.set_title(f"Accuracy Heatmap: {model}", fontsize=14)

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/heatmap_{model.replace('.', '_')}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/heatmap_{model.replace('.', '_')}.png")


def compute_statistics(df: pd.DataFrame):
    """Compute statistical tests on results."""
    llm_models = [m for m in df["model"].unique() if "baseline" not in m]

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    for model in llm_models:
        model_df = df[df["model"] == model]

        # Correlation between character count and accuracy
        char_counts = model_df.groupby("num_characters")["correct"].mean()
        correlation, p_value = stats.pearsonr(char_counts.index, char_counts.values)

        print(f"\n{model}:")
        print(f"  Correlation (chars vs accuracy): r = {correlation:.3f}, p = {p_value:.4f}")

        # Linear regression slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(char_counts.index, char_counts.values)
        print(f"  Linear regression: slope = {slope:.4f}, R² = {r_value**2:.3f}")

        # Compare low (2-5) vs high (15-20) character performance
        low_char = model_df[model_df["num_characters"] <= 5]["correct"]
        high_char = model_df[model_df["num_characters"] >= 15]["correct"]

        if len(low_char) > 0 and len(high_char) > 0:
            t_stat, t_pvalue = stats.ttest_ind(low_char, high_char)
            effect_size = (low_char.mean() - high_char.mean()) / np.sqrt((low_char.std()**2 + high_char.std()**2) / 2)
            print(f"  Low (2-5) vs High (15-20) chars: t = {t_stat:.2f}, p = {t_pvalue:.4f}, Cohen's d = {effect_size:.3f}")
            print(f"    Low char accuracy: {low_char.mean():.3f}, High char accuracy: {high_char.mean():.3f}")


def main():
    """Generate all visualizations."""
    print("Loading results...")
    df, analysis = load_results()

    print("Generating visualizations...")
    plot_accuracy_vs_characters(df)
    plot_accuracy_by_question_type(df)
    plot_accuracy_by_actions(df)
    plot_heatmap(df)

    compute_statistics(df)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
