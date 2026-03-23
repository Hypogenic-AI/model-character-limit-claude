# Datasets for Character/Entity Tracking Research

This directory contains datasets for studying how many characters an LLM can keep track of in narratives.

## Summary

| Dataset | Purpose | Size | Format | Location |
|---------|---------|------|--------|----------|
| Boxes (Kim & Schuster 2023) | Entity state tracking in closed-world | ~90k examples (v1 original) | JSONL | `boxes/` |
| OpenToM | Theory of mind, character location/belief | 13,708 examples | JSON | `opentom/` |
| ProPara (allenai) | Procedural entity state tracking | ~488 paragraphs, 81k annotations | JSON/TSV | `propara/` |
| SimpleToM (allenai) | Explicit + applied theory of mind | 1,147 stories × 3 QA types | JSONL | `simpletom/` |
| NarrativeQA (DeepMind) | Story reading comprehension | 32k train / 10.5k test | JSONL | `narrativeqa/` |
| Theory of Mind (grimulkan) | ToM instruction-following | 539 examples | JSONL | `theory_of_mind_grimulkan/` |
| character_tracking_synthetic | Primary synthetic experiment set | 90 examples | JSON | `character_tracking_synthetic.json` |

**Note:** CHATTER (character attribution, arxiv 2411.05227) does not have a public data release; its GitHub repo (usc-sail/mica-character-attribution) contains only crawling/processing scripts. See below for details.

---

## Dataset 1: Boxes (Kim & Schuster, ACL 2023)

### Overview
- **Paper**: "Entity Tracking in Language Models" (ACL 2023)
- **Task**: Track which objects are in which boxes after a sequence of move/put/remove operations
- **Source code**: `/workspaces/model-character-limit-claude/code/entity-tracking-lms/`
- **Format**: JSONL with fields: `sentence`, `sentence_masked`, `masked_content`, `sample_id`, `numops`

### Contents
- `boxes/boxes-v1-original/` — original release (password-extracted from ZIP, password: `iamnotaLM`)
  - `t5_boxes_nso_exp2_max3/` — main split: 90k train, 20k dev, 90k test
  - `t5_boxes_nso_exp2_max3_alt_forms_train/` — alternative surface forms
  - `t5_boxes_nso_exp2_max3_move_contents/` — "move contents" operation variant
  - `few_shot_boxes_nso_exp2_max3/` — few-shot format
- `boxes/` (root) — freshly generated split (2200 samples, max_items_per_box=1, 7 boxes, 10 ops)

### Key parameters
- 7 boxes, up to 3 items per box (original v1), 10 operations per sequence
- Splits by number of operations: 0–10 (tests compositional generalization)
- Also varies: object vocabulary (in/out of BNC), alternative surface forms

### Re-generation
```bash
cd /workspaces/model-character-limit-claude/code/entity-tracking-lms
source /workspaces/model-character-limit-claude/.venv/bin/activate
python src/dataset_generation/generate_boxes_data.py \
    --max_items_per_box 1 \
    --num_samples 2200 \
    --output_dir /workspaces/model-character-limit-claude/datasets/boxes/ \
    --object_vocabulary_file data/objects_with_bnc_frequency.csv \
    --disjoint_object_vocabulary_file data/objects_not_in_bnc.csv
```

### Citation
```bibtex
@inproceedings{kim-schuster-2023-entity,
    title = "Entity Tracking in Language Models",
    author = "Kim, Najoung and Schuster, Sebastian",
    booktitle = "Proceedings of ACL 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.acl-long.213"
}
```

---

## Dataset 2: OpenToM

### Overview
- **Paper**: "OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models" (2024)
- **Task**: Character location tracking + belief reasoning (first-order and second-order ToM)
- **Source**: `/workspaces/model-character-limit-claude/code/OpenToM/data/` (copied here)
- **Format**: JSON

### Contents
- `opentom.json` — 13,708 QA examples (standard narratives)
- `opentom_long.json` — longer narrative version
- `opentom_data/` — broken down by question type:
  - `location_cg_fo.json` / `location_cg_so.json` — coarse-grained location (first/second order)
  - `location_fg_fo_new.json` / `location_fg_so_new.json` — fine-grained location
  - `multihop_fo.json` / `multihop_so.json` — multi-hop reasoning
  - `attitude.json` — character attitude questions
  - `meta_data.json` / `meta_data_long.json` — story metadata

### Example structure
```json
{
  "plot": "Diego entered the patio. Amir entered the patio...",
  "plot_info": {"mover": "Diego", "eoi": "scarf", "original_place": "basket", ...},
  "preferences": {"mover": "...", "observer": "..."},
  "personality": "Diego is an inconsiderate person.",
  "questions": [...]
}
```

### HuggingFace
Dataset `hkust-nlp/opentom` is not currently accessible on the Hub. Use local copy.

---

## Dataset 3: ProPara (AllenAI, NAACL 2018 / EMNLP 2018)

### Overview
- **Paper**: "Tracking State Changes in Procedural Text" (NAACL 2018)
- **Task**: Track entity existence and location through science process paragraphs
- **Source**: Downloaded from https://github.com/allenai/propara
- **Size**: 488 paragraphs, ~81k state annotations
- **Format**: JSON (JSONL, one paragraph per line) and TSV

### Contents
- `propara/raw/` — raw files from GitHub
  - `grids.v1.train.json` / `grids.v1.dev.json` / `grids.v1.test.json` — EMNLP18 format
  - `grids.v1.train.tsv` / `grids.v1.dev.tsv` / `grids.v1.test.tsv` — TSV format
  - `gold-full-grids.v3.tsv` — NAACL18 gold annotations
- `propara/huggingface/` — task-formatted versions from HuggingFace
  - `task1566_propara_structured_text_generation_*.jsonl` — structured generation
  - `task1567_propara_question_generation_*.jsonl` — question generation
  - `task1568_propara_classification_*.jsonl` — state change classification

### Example structure
```json
{
  "para_id": "37",
  "sentence_texts": ["A plant or animal dies...", "Is buried in mud and silt.", ...],
  "participants": ["plant; animal", "soft tissues", "bones", "mineral", "fossils"],
  "states": [["watery environment", "watery environment", ...], ...]
}
```

### Download instructions (refresh)
```bash
mkdir -p datasets/propara/raw
for f in grids.v1.train.json grids.v1.dev.json grids.v1.test.json \
          grids.v1.train.tsv grids.v1.dev.tsv grids.v1.test.tsv; do
    curl -L "https://raw.githubusercontent.com/allenai/propara/master/data/emnlp18/$f" \
         -o datasets/propara/raw/$f
done
curl -L "https://raw.githubusercontent.com/allenai/propara/master/data/naacl18/gold-full-grids.v3.tsv" \
     -o datasets/propara/raw/gold-full-grids.v3.tsv
```

### Citation
```bibtex
@inproceedings{mishra2018tracking,
    title = "Tracking State Changes in Procedural Text: a Challenge Dataset and Models for Process Paragraph Comprehension",
    author = "Mishra, Bhavana Dalvi and Tandon, Niket and Bhagavatula, Chandra and Clark, Peter",
    booktitle = "NAACL 2018",
    year = "2018"
}
```

---

## Dataset 4: SimpleToM (AllenAI, 2024)

### Overview
- **Paper**: "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs" (2024)
- **Task**: Theory of mind — mental state, behavior prediction, judgment
- **Source**: `allenai/SimpleToM` on HuggingFace
- **Size**: 1,147 stories × 3 QA types = 3,441 total QA examples
- **Format**: JSONL

### Contents
- `simpletom/mental-state-qa_test.jsonl` — information awareness questions
- `simpletom/behavior-qa_test.jsonl` — future behavior prediction
- `simpletom/judgment-qa_test.jsonl` — reasonableness of behavior
- `simpletom/story-data_test.jsonl` — full story metadata

### Download instructions (refresh)
```python
from datasets import load_dataset
import json
for config in ['mental-state-qa', 'behavior-qa', 'judgment-qa', 'story-data']:
    ds = load_dataset('allenai/SimpleToM', config)
    for split, data in ds.items():
        with open(f'datasets/simpletom/{config}_{split}.jsonl', 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
```

### Citation
```bibtex
@article{wilf2024simpletom,
    title = "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs",
    author = "Wilf, Alex and others",
    year = "2024",
    url = "https://arxiv.org/abs/2410.13648"
}
```

---

## Dataset 5: NarrativeQA (DeepMind, 2018)

### Overview
- **Paper**: "The NarrativeQA Reading Comprehension Challenge" (TACL 2018)
- **Task**: Reading comprehension over full books and movie scripts
- **Source**: `deepmind/narrativeqa` on HuggingFace
- **Size**: 32,747 train / 3,461 validation / 10,557 test QA pairs across 1,567 documents
- **Format**: JSONL (Q&A pairs; full document texts not stored locally due to size)

### Contents
- `narrativeqa/train.jsonl`, `test.jsonl`, `validation.jsonl`
- Fields: `document_id`, `document_kind` (book/movie), `question`, `answers`

### Note on characters
NarrativeQA questions frequently ask about character actions, relationships, and motivations. Useful as a naturalistic benchmark for character tracking.

### Download instructions (refresh)
```python
from datasets import load_dataset
import json
ds = load_dataset('deepmind/narrativeqa')
for split, data in ds.items():
    with open(f'datasets/narrativeqa/{split}.jsonl', 'w') as f:
        for ex in data:
            row = {
                'document_id': ex['document']['id'],
                'document_kind': ex['document']['kind'],
                'question': ex['question'],
                'answers': ex['answers'],
            }
            f.write(json.dumps(row) + '\n')
```

### Citation
```bibtex
@article{kovcisky2018narrativeqa,
    title = "The NarrativeQA Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Schwarz, Jonathan and Blunsom, Phil and others",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    pages = "317--328",
    year = "2018"
}
```

---

## Dataset 6: Theory of Mind (grimulkan)

### Overview
- **Task**: Theory of mind instruction-following (single-turn)
- **Source**: `grimulkan/theory-of-mind` on HuggingFace
- **Size**: 539 examples
- **Format**: JSONL (instruction/input/response triples)

### Contents
- `theory_of_mind_grimulkan/train.jsonl`

### Download instructions (refresh)
```python
from datasets import load_dataset
import json
ds = load_dataset('grimulkan/theory-of-mind')
for split, data in ds.items():
    with open(f'datasets/theory_of_mind_grimulkan/{split}.jsonl', 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')
```

---

## Dataset 7: character_tracking_synthetic.json (existing)

Previously generated synthetic dataset for primary character tracking experiments.
See original README content above for structure and usage.

---

## CHATTER (Not Available Publicly)

**Paper**: "CHATTER: A Character Attribution Dataset for Narrative Understanding" (arXiv 2411.05227, 2024)

CHATTER labels 88,148 character-attribute pairs across 2,998 characters and 660 movies. A validated subset called ChatterEval serves as an evaluation benchmark.

**Status**: The GitHub repository (https://github.com/usc-sail/mica-character-attribution) contains only data collection and processing scripts. The actual dataset (screenplay text + annotations) is not publicly released and requires running the crawling pipeline with licensed screenplay data.

---

## Samples

Quick-reference examples (5 instances each) are in `samples/`:
- `boxes_sample.json` — Boxes entity tracking examples
- `opentom_sample.json` — OpenToM theory of mind examples
- `propara_sample.json` — ProPara procedural paragraphs
- `simpletom_sample.json` — SimpleToM mental-state QA
- `narrativeqa_sample.json` — NarrativeQA question-answer pairs

---

## Experimental Design Notes

### For studying "how many characters can a model keep track of"

| Dataset | Characters per story | Relevance |
|---------|---------------------|-----------|
| Boxes | N/A (objects in boxes) | Entity state tracking; scales with box/object count |
| OpenToM | 2–5 typically | Character location + belief tracking in short narratives |
| ProPara | 3–8 entities per paragraph | State change (exist/location) over process steps |
| SimpleToM | 1–2 (dyadic) | Minimal characters; tests ToM inference quality |
| NarrativeQA | 5–50+ | Full-length stories; character-centric QA |

### Recommended experiment pipeline
1. **Controlled scaling**: Use Boxes (vary box count) or generate synthetic narratives (vary character count) to find the break point
2. **Naturalistic baseline**: Use ProPara and OpenToM as semi-naturalistic benchmarks
3. **Full narrative**: Use NarrativeQA for real-world upper bound

---

## Installation

```bash
source /workspaces/model-character-limit-claude/.venv/bin/activate
uv pip install datasets huggingface_hub
```
