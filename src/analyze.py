"""
Analysis and visualization of character tracking experiment results.
Produces degradation curves, error type analysis, and position effects.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

RESULTS_DIR = "/workspaces/model-character-limit-claude/results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_results():
    """Load all experiment results."""
    # Main experiment (5 stories per count)
    with open(f"{RESULTS_DIR}/all_results.json") as f:
        main_results = json.load(f)

    # Extended experiment (3 stories per count, includes 48 and 64)
    with open(f"{RESULTS_DIR}/extended_results.json") as f:
        extended_results = json.load(f)

    # Stress test (4 state changes)
    with open(f"{RESULTS_DIR}/stress_results.json") as f:
        stress_results = json.load(f)

    # Baselines
    with open(f"{RESULTS_DIR}/baselines.json") as f:
        baselines = json.load(f)

    with open(f"{RESULTS_DIR}/extended_baselines.json") as f:
        ext_baselines = json.load(f)

    return main_results, extended_results, stress_results, baselines, ext_baselines


def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.RandomState(42)
    means = []
    data = np.array(data)
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper


def analyze_main_results(results):
    """Analyze the main experiment results."""
    df = pd.DataFrame(results)

    print("=" * 70)
    print("MAIN EXPERIMENT ANALYSIS (2 state changes per character)")
    print("=" * 70)

    # Overall accuracy by model and character count
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        print(f"\n--- {model} ---")
        for nc in sorted(mdf["num_characters"].unique()):
            ncdf = mdf[mdf["num_characters"] == nc]
            correct = ncdf["is_correct"].values.astype(float)
            mean, lo, hi = bootstrap_ci(correct)
            n = len(correct)
            print(f"  {nc:2d} characters: {mean:.1%} [{lo:.1%}, {hi:.1%}] (n={n})")

    return df


def analyze_extended(results):
    """Analyze extended results including 48 and 64 characters."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("EXTENDED EXPERIMENT (includes 48 and 64 characters)")
    print("=" * 70)

    summary = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        print(f"\n--- {model} ---")
        model_summary = {}
        for nc in sorted(mdf["num_characters"].unique()):
            ncdf = mdf[mdf["num_characters"] == nc]
            correct = ncdf["is_correct"].values.astype(float)
            mean, lo, hi = bootstrap_ci(correct)
            print(f"  {nc:2d} characters: {mean:.1%} [{lo:.1%}, {hi:.1%}] (n={len(correct)})")
            model_summary[nc] = {"mean": mean, "lo": lo, "hi": hi, "n": len(correct)}
        summary[model] = model_summary

    return df, summary


def analyze_errors(results):
    """Analyze error types across conditions."""
    df = pd.DataFrame(results)
    errors = df[~df["is_correct"]]

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    for model in df["model"].unique():
        merr = errors[errors["model"] == model]
        if len(merr) == 0:
            print(f"\n--- {model}: No errors ---")
            continue

        print(f"\n--- {model} ({len(merr)} errors total) ---")

        # Error types
        type_counts = merr["error_type"].value_counts()
        total_errors = len(merr)
        for etype, count in type_counts.items():
            print(f"  {etype}: {count} ({count/total_errors:.1%})")

        # Error types by character count
        print(f"\n  Error types by character count:")
        for nc in sorted(merr["num_characters"].unique()):
            ncerr = merr[merr["num_characters"] == nc]
            types = ncerr["error_type"].value_counts().to_dict()
            nc_total = len(df[(df["model"] == model) & (df["num_characters"] == nc)])
            nc_err_count = len(ncerr)
            print(f"    {nc:2d} chars: {nc_err_count} errors / {nc_total} questions "
                  f"({nc_err_count/nc_total:.1%}) — {dict(types)}")

    return errors


def analyze_position_effect(results):
    """Analyze whether character introduction order affects accuracy."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("POSITION EFFECT ANALYSIS")
    print("=" * 70)

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        # Only look at stories with enough characters
        mdf_big = mdf[mdf["num_characters"] >= 16]
        if len(mdf_big) == 0:
            continue

        print(f"\n--- {model} (stories with ≥16 characters) ---")

        # Bin by introduction order quartile
        mdf_big = mdf_big.copy()
        mdf_big["order_quartile"] = pd.qcut(
            mdf_big["introduction_order"],
            q=4,
            labels=["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"],
            duplicates="drop"
        )

        for q in ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]:
            qdf = mdf_big[mdf_big["order_quartile"] == q]
            if len(qdf) > 0:
                acc = qdf["is_correct"].mean()
                print(f"  {q}: {acc:.1%} (n={len(qdf)})")

        # Correlation
        corr, pval = stats.spearmanr(mdf_big["introduction_order"], mdf_big["is_correct"].astype(float))
        print(f"  Spearman correlation (order vs correct): r={corr:.3f}, p={pval:.4f}")


def analyze_attribute_type(results):
    """Analyze accuracy by attribute type (location vs possession)."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("ATTRIBUTE TYPE ANALYSIS")
    print("=" * 70)

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        print(f"\n--- {model} ---")
        for attr in ["location", "possession"]:
            adf = mdf[mdf["attribute"] == attr]
            acc = adf["is_correct"].mean()
            print(f"  {attr}: {acc:.1%} (n={len(adf)})")

            for nc in sorted(adf["num_characters"].unique()):
                ncdf = adf[adf["num_characters"] == nc]
                print(f"    {nc:2d} chars: {ncdf['is_correct'].mean():.1%}")


def plot_degradation_curve(extended_df, summary):
    """Plot accuracy vs character count for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"gpt-4.1": "#2196F3", "gpt-4.1-mini": "#FF9800"}
    markers = {"gpt-4.1": "o", "gpt-4.1-mini": "s"}

    for model, model_data in summary.items():
        xs = sorted(model_data.keys())
        means = [model_data[x]["mean"] * 100 for x in xs]
        los = [model_data[x]["lo"] * 100 for x in xs]
        his = [model_data[x]["hi"] * 100 for x in xs]

        ax.plot(xs, means, marker=markers.get(model, "o"), color=colors.get(model, "gray"),
                linewidth=2, markersize=8, label=model, zorder=3)
        ax.fill_between(xs, los, his, alpha=0.2, color=colors.get(model, "gray"))

    # Add baselines
    ax.axhline(y=5.7, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.axhline(y=28.4, color='gray', linestyle=':', alpha=0.5, label='Initial-state baseline')

    ax.set_xlabel("Number of Characters", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Character Tracking Accuracy vs. Number of Characters\n(2 state changes per character)", fontsize=14)
    ax.set_xticks([2, 4, 8, 16, 32, 48, 64])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{PLOTS_DIR}/degradation_curve.png")
    plt.close()
    print(f"\nSaved: {PLOTS_DIR}/degradation_curve.png")


def plot_error_types(extended_df):
    """Plot error type distribution by character count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, model in enumerate(extended_df["model"].unique()):
        ax = axes[idx]
        mdf = extended_df[extended_df["model"] == model]
        errors = mdf[~mdf["is_correct"]]

        if len(errors) == 0:
            ax.set_title(f"{model}\n(no errors)")
            continue

        char_counts = sorted(errors["num_characters"].unique())
        error_types = ["confusion", "forgetting", "missing", "other"]
        colors_map = {"confusion": "#E53935", "forgetting": "#FFA726", "missing": "#78909C", "other": "#9E9E9E"}

        bottoms = {nc: 0 for nc in char_counts}
        for etype in error_types:
            heights = []
            for nc in char_counts:
                nc_total = len(mdf[mdf["num_characters"] == nc])
                nc_err = len(errors[(errors["num_characters"] == nc) & (errors["error_type"] == etype)])
                rate = nc_err / nc_total * 100 if nc_total > 0 else 0
                heights.append(rate)

            bars = ax.bar([str(nc) for nc in char_counts], heights,
                         bottom=[bottoms[nc] for nc in char_counts],
                         color=colors_map.get(etype, "gray"), label=etype, alpha=0.85)
            for nc, h in zip(char_counts, heights):
                bottoms[nc] += h

        ax.set_xlabel("Number of Characters", fontsize=12)
        ax.set_title(f"{model}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel("Error Rate (%)", fontsize=12)
    fig.suptitle("Error Type Distribution by Character Count", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/error_types.png")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/error_types.png")


def plot_stress_comparison(main_df, stress_results):
    """Plot comparison of 2 vs 4 state changes."""
    stress_df = pd.DataFrame(stress_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, model in enumerate(["gpt-4.1-mini", "gpt-4.1"]):
        ax = axes[idx]

        # Main (2 changes)
        mdf_main = main_df[main_df["model"] == model]
        char_counts_main = sorted(mdf_main["num_characters"].unique())
        acc_main = [mdf_main[mdf_main["num_characters"] == nc]["is_correct"].mean() * 100 for nc in char_counts_main]

        # Stress (4 changes)
        mdf_stress = stress_df[stress_df["model"] == model]
        char_counts_stress = sorted(mdf_stress["num_characters"].unique())
        acc_stress = [mdf_stress[mdf_stress["num_characters"] == nc]["is_correct"].mean() * 100 for nc in char_counts_stress]

        ax.plot(char_counts_main, acc_main, 'o-', color="#2196F3", linewidth=2, markersize=8, label="2 changes/char")
        ax.plot(char_counts_stress, acc_stress, 's--', color="#E53935", linewidth=2, markersize=8, label="4 changes/char")

        ax.set_xlabel("Number of Characters", fontsize=12)
        ax.set_title(f"{model}", fontsize=13)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    fig.suptitle("Effect of State Changes per Character on Tracking Accuracy", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/stress_comparison.png")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/stress_comparison.png")


def plot_attribute_comparison(extended_df):
    """Plot accuracy by attribute type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in extended_df["model"].unique():
        for attr in ["location", "possession"]:
            mdf = extended_df[(extended_df["model"] == model) & (extended_df["attribute"] == attr)]
            char_counts = sorted(mdf["num_characters"].unique())
            accs = [mdf[mdf["num_characters"] == nc]["is_correct"].mean() * 100 for nc in char_counts]

            style = "-" if attr == "location" else "--"
            color = "#2196F3" if "4.1-mini" not in model else "#FF9800"
            marker = "o" if attr == "location" else "s"
            ax.plot(char_counts, accs, f"{marker}{style}", color=color,
                    linewidth=2, markersize=7, label=f"{model} ({attr})", alpha=0.8)

    ax.set_xlabel("Number of Characters", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Tracking Accuracy by Attribute Type", fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/attribute_comparison.png")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/attribute_comparison.png")


def compute_statistics(extended_df):
    """Compute statistical tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    for model in extended_df["model"].unique():
        mdf = extended_df[extended_df["model"] == model]
        print(f"\n--- {model} ---")

        # Spearman correlation: character count vs accuracy
        corr, pval = stats.spearmanr(mdf["num_characters"], mdf["is_correct"].astype(float))
        print(f"  Spearman (char_count vs accuracy): r={corr:.4f}, p={pval:.6f}")

        # Compare 2-char vs 32-char accuracy (McNemar-like with proportions test)
        acc_2 = mdf[mdf["num_characters"] == 2]["is_correct"].values
        acc_32 = mdf[mdf["num_characters"] == 32]["is_correct"].values
        if len(acc_2) > 0 and len(acc_32) > 0:
            # Two-proportion z-test
            p1 = acc_2.mean()
            p2 = acc_32.mean()
            n1 = len(acc_2)
            n2 = len(acc_32)
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                z = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                print(f"  2-char ({p1:.1%}) vs 32-char ({p2:.1%}): z={z:.3f}, p={p_value:.6f}")


def generate_summary_table(main_df, extended_df, stress_df):
    """Generate summary table for the report."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    stress = pd.DataFrame(stress_df)

    print(f"\n{'Model':<20} {'N_chars':<10} {'Accuracy':<12} {'95% CI':<20} {'N':<6}")
    print("-" * 68)

    for model in extended_df["model"].unique():
        mdf = extended_df[extended_df["model"] == model]
        for nc in sorted(mdf["num_characters"].unique()):
            ncdf = mdf[mdf["num_characters"] == nc]
            correct = ncdf["is_correct"].values.astype(float)
            mean, lo, hi = bootstrap_ci(correct)
            print(f"{model:<20} {nc:<10} {mean:.1%}{'':>5} [{lo:.1%}, {hi:.1%}]{'':>3} {len(correct):<6}")
        print()


if __name__ == "__main__":
    main_results, extended_results, stress_results, baselines, ext_baselines = load_results()

    # Analyze
    main_df = analyze_main_results(main_results)
    extended_df, summary = analyze_extended(extended_results)
    errors = analyze_errors(extended_results)
    analyze_position_effect(extended_results)
    analyze_attribute_type(extended_results)
    compute_statistics(extended_df)

    # Plots
    plot_degradation_curve(extended_df, summary)
    plot_error_types(extended_df)
    plot_stress_comparison(main_df, stress_results)
    plot_attribute_comparison(extended_df)

    # Summary
    generate_summary_table(main_df, extended_df, stress_results)

    print("\n\nAnalysis complete!")
