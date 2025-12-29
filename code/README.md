# Code Repositories

This directory contains cloned repositories relevant to the character tracking research.

## Repositories

### 1. entity-tracking-lms
- **URL**: https://github.com/sebschu/entity-tracking-lms
- **Paper**: "Entity Tracking in Language Models" (Kim & Schuster, ACL 2023)
- **Purpose**: Entity tracking task and evaluation framework
- **Location**: `code/entity-tracking-lms/`

#### Key Files
- `src/`: Main source code for entity tracking experiments
- `data/`: Data generation scripts for box-moving task
- `scripts/`: Evaluation and training scripts

#### Usage
```bash
cd entity-tracking-lms
pip install -r requirements.txt
# See README.md in the repo for detailed instructions
```

---

### 2. ruler-benchmark
- **URL**: https://github.com/NVIDIA/RULER
- **Paper**: "RULER: What's the Real Context Size of Your Long-Context Language Models?" (COLM 2024)
- **Purpose**: Comprehensive long-context LLM evaluation including variable tracking
- **Location**: `code/ruler-benchmark/`

#### Key Files
- `scripts/`: Task generation and evaluation scripts
- `data/`: Synthetic task configurations
- `synthetic/`: Task-specific generation code (including variable tracking)

#### Relevant Tasks for Character Tracking
- **Variable Tracking (VT)**: Multi-hop entity/variable tracking
- **NIAH variants**: Tests retrieval across context positions

#### Usage
```bash
cd ruler-benchmark
pip install -r requirements.txt
# Generate variable tracking task:
python scripts/generate_variable_tracking.py --num_hops 4 --context_length 4096
```

---

### 3. lost-in-the-middle
- **URL**: https://github.com/nelson-liu/lost-in-the-middle
- **Paper**: "Lost in the Middle: How Language Models Use Long Contexts" (TACL 2024)
- **Purpose**: Position-based retrieval experiments and analysis
- **Location**: `code/lost-in-the-middle/`

#### Key Files
- `src/`: Experiment code
- `scripts/`: Evaluation scripts
- `data/`: Multi-document QA and key-value retrieval data

#### Relevance
Demonstrates that models struggle with information in the middle of contexts - important for understanding character tracking failures.

---

## How to Use These Repositories

### For Baseline Comparisons

1. **Entity Tracking Baseline**: Use `entity-tracking-lms` to compare against the established entity tracking benchmark (box-moving task)

2. **Long-Context Evaluation**: Use `ruler-benchmark` to test variable tracking across different context lengths

3. **Position Effects**: Use `lost-in-the-middle` to understand if character tracking failures are position-dependent

### Recommended Workflow

```python
# Example: Use RULER's variable tracking task generation
import sys
sys.path.append('code/ruler-benchmark/synthetic')
# Import their task generation utilities

# Example: Use entity-tracking-lms evaluation metrics
sys.path.append('code/entity-tracking-lms/src')
# Import their evaluation code
```

---

## Integration with Our Research

Our hypothesis about character tracking limits can be tested by:

1. **Varying the number of entities** (like our synthetic dataset does)
2. **Measuring where accuracy drops** (using RULER-style metrics)
3. **Checking for position effects** (using lost-in-the-middle analysis)

The code in these repositories provides:
- Proven evaluation frameworks
- Data generation utilities
- Baseline implementations for comparison
