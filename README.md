# How Many Characters Can a Language Model Keep Track Of?

Research project investigating the limits of character tracking in frontier language models using controlled synthetic narratives.

## Key Findings

- **Perfect tracking up to 16 characters**: Both GPT-4.1 and GPT-4.1-mini achieve 100% accuracy with up to 16 characters
- **Graceful degradation, not a cliff**: Accuracy drops gradually to ~93% at 64 characters — no sharp threshold
- **Confusion, not forgetting**: 93.9% of errors are confusion errors (swapping attributes between characters), suggesting overlapping distributed representations
- **Smaller model wins**: GPT-4.1-mini (95.3%) outperforms GPT-4.1 (93.6%) overall, a counterintuitive finding
- **State change complexity matters**: Doubling state changes from 2→4 per character drops GPT-4.1 from 86.9% to 84.4% at 32 characters

See [REPORT.md](REPORT.md) for the full research report with statistical analysis and visualizations.

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy matplotlib pandas scipy tqdm

# Run experiments
export OPENAI_API_KEY=your_key
python src/experiment.py           # Main experiment (2-32 chars)
python src/experiment_extended.py  # Extended (up to 64 chars + stress test)
python src/analyze.py              # Analysis and plots
```

## Project Structure

```
src/
├── story_generator.py      # Synthetic narrative generator (2-64 characters)
├── experiment.py            # Main experiment runner
├── experiment_extended.py   # Extended experiment (48, 64 chars + stress test)
└── analyze.py               # Statistical analysis and visualization
results/
├── plots/                   # Degradation curves, error analysis plots
├── all_results.json         # Raw experimental results
└── extended_results.json    # Extended experiment results
planning.md                  # Research plan and methodology
REPORT.md                    # Full research report
literature_review.md         # Literature review
resources.md                 # Resource catalog
```
