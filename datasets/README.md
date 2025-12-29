# Datasets for Character Tracking Research

This directory contains datasets for studying LLM character tracking limits.

## Summary

| Dataset | Purpose | Size | Format |
|---------|---------|------|--------|
| character_tracking_synthetic | Primary experiment dataset | 90 examples | JSON |
| bAbI Tasks | Entity tracking baseline | 10K train/1K test per task | TXT |
| NarrativeQA samples | Story QA reference | 10 samples | JSON |

## Dataset 1: Character Tracking Synthetic Dataset

### Overview
- **Source**: Generated with `character_tracking_generator.py`
- **Size**: 90 examples covering various configurations
- **Format**: JSON
- **Task**: Track multiple characters' states through a narrative

### Configuration
The dataset systematically varies:
- **Number of characters**: 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
- **Number of actions**: 5, 10, 20
- **3 trials** per configuration for statistical reliability

### Data Structure
```json
{
  "story": "Full narrative text",
  "sentences": ["List", "of", "sentences"],
  "characters": ["Alice", "Bob", ...],
  "num_characters": 5,
  "num_actions": 10,
  "final_states": {
    "Alice": {"location": "kitchen", "holding": "book", "mood": "happy"}
  },
  "questions": [
    {"question": "Where is Alice?", "answer": "kitchen", "type": "location"}
  ]
}
```

### Question Types
1. **Location**: "Where is {character}?"
2. **Mood**: "How does {character} feel?"
3. **Holding**: "What is {character} holding?"

### Usage
```python
import json
with open("datasets/character_tracking_synthetic.json") as f:
    data = json.load(f)

# Access examples
for example in data["examples"]:
    print(f"Characters: {example['num_characters']}")
    print(f"Story: {example['story'][:200]}...")
```

---

## Dataset 2: bAbI Tasks (Facebook Research)

### Overview
- **Source**: Facebook AI Research
- **Size**: 10,000 training / 1,000 test examples per task
- **Format**: Plain text
- **Location**: `tasks_1-20_v1-2/en/`

### Relevant Tasks for Entity Tracking
| Task | Name | Description |
|------|------|-------------|
| qa1 | Single Supporting Fact | Location tracking (1 hop) |
| qa2 | Two Supporting Facts | Object+location tracking (2 hops) |
| qa3 | Three Supporting Facts | Multi-hop reasoning |
| qa5 | Three Argument Relations | Complex relations |
| qa11 | Basic Coreference | Pronoun resolution |
| qa13 | Compound Coreference | Complex coreference |

### Download Instructions

**Already downloaded locally:**
```bash
# Data is in datasets/tasks_1-20_v1-2/en/
```

**To download fresh:**
```bash
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xzf tasks_1-20_v1-2.tar.gz
```

### Format
Each line contains: `line_number text` or `line_number question \t answer \t supporting_facts`

Example:
```
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?     bathroom    1
```

### Usage
```python
def parse_babi(filename):
    stories = []
    current = {"context": [], "qa": []}
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0])
            if idx == 1 and current["context"]:
                stories.append(current)
                current = {"context": [], "qa": []}
            text = " ".join(parts[1:])
            if "\t" in text:
                q, a, _ = text.split("\t")
                current["qa"].append((q, a))
            else:
                current["context"].append(text)
    return stories
```

---

## Dataset 3: NarrativeQA Samples

### Overview
- **Source**: DeepMind (Hugging Face)
- **Size**: 10 sample summaries (full dataset: 1,567 stories)
- **Format**: JSON
- **Task**: Reading comprehension on narratives

### Download Instructions

**Full dataset (via HuggingFace):**
```python
from datasets import load_dataset
narr = load_dataset("deepmind/narrativeqa")
```

### Notes
- Full stories are very long (100k+ words)
- Useful for understanding character-centric question types
- Questions often require reasoning about character relationships

---

## Experimental Design Recommendations

### Primary Experiments
Use `character_tracking_synthetic.json` for controlled experiments:
1. Fix number of actions, vary character count
2. Measure accuracy degradation as characters increase
3. Identify "breaking point" where model performance drops

### Baseline Comparisons
Use bAbI tasks to establish baseline performance:
- Task 1: Basic location tracking
- Task 2: Object+location tracking
- Compare synthetic dataset performance to bAbI benchmarks

### Evaluation Metrics
- **Accuracy by character count**: What % of questions are correct?
- **Per-character accuracy**: Does accuracy degrade for specific characters?
- **Question type breakdown**: Location vs. mood vs. holding

---

## Citation

If using bAbI tasks:
```bibtex
@article{weston2015towards,
  title={Towards AI-complete question answering: A set of prerequisite toy tasks},
  author={Weston, Jason and Bordes, Antoine and Chopra, Sumit and Rush, Alexander M and van Merri{\"e}nboer, Bart and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1502.05698},
  year={2015}
}
```

If using NarrativeQA:
```bibtex
@article{kovcisky2018narrativeqa,
  title={The NarrativeQA reading comprehension challenge},
  author={Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Schwarz, Jonathan and Blunsom, Phil and Dyer, Chris and Hermann, Karl Moritz and Melis, G{\'a}bor and Grefenstette, Edward},
  journal={Transactions of the Association for Computational Linguistics},
  volume={6},
  pages={317--328},
  year={2018}
}
```
