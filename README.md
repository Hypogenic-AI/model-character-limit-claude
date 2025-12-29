# Character Tracking Limits in Language Models

## Overview

This research project investigates how many characters language models can keep track of in-context. We systematically tested GPT-4.1 and GPT-3.5-turbo on synthetic narratives with 2-20 characters, measuring their ability to track character states (location, mood, possessions).

## Key Findings

- **GPT-4.1 achieves 99.95% accuracy** across all configurations, showing no observable character tracking limit up to 20 characters
- **GPT-3.5-turbo achieves 81.4% accuracy** with no degradation as character count increases
- **Question type matters**: "Holding" questions are 19% harder than location/mood questions for GPT-3.5-turbo
- **No capacity cliff found**: Both models maintain consistent performance from 2 to 20 characters

## Project Structure

```
.
├── REPORT.md                    # Full research report with findings
├── README.md                    # This file
├── planning.md                  # Research plan
├── pyproject.toml               # Project dependencies
├── src/
│   ├── experiment.py            # Main experiment code
│   └── visualize.py             # Visualization generation
├── datasets/
│   ├── character_tracking_synthetic.json  # Primary dataset (90 examples)
│   └── tasks_1-20_v1-2/         # bAbI tasks for reference
├── results/
│   ├── raw_results.csv          # All predictions and ground truth
│   ├── analysis.json            # Computed statistics
│   └── figures/                 # Visualization plots
├── papers/                      # Reference papers
├── code/                        # Baseline implementations
└── literature_review.md         # Background literature review
```

## Quick Start

```bash
# Set up environment
uv venv
source .venv/bin/activate
uv add numpy pandas matplotlib openai httpx tqdm scipy

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiment
python src/experiment.py

# Generate visualizations
python src/visualize.py
```

## Dataset

The synthetic dataset (`datasets/character_tracking_synthetic.json`) contains:
- 90 stories with varying configurations
- Character counts: 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
- Action counts: 5, 10, 20
- 3 trials per configuration

Each story has questions about each character's:
- **Location**: "Where is Alice?" → "kitchen"
- **Mood**: "How does Alice feel?" → "happy"
- **Holding**: "What is Alice holding?" → "book" or "nothing"

## Results Summary

| Model | Overall Accuracy |
|-------|-----------------|
| GPT-4.1 | 99.95% |
| GPT-3.5-turbo | 81.4% |
| First-State Baseline | 78.2% |
| Random Baseline | 12.7% |

See `REPORT.md` for detailed analysis and visualizations.

## References

- Kim & Schuster (2023). Entity Tracking in Language Models. ACL.
- Liu et al. (2024). Lost in the Middle. TACL.
- Hsieh et al. (2024). RULER Benchmark. COLM.

## License

Research code for experimental purposes.
