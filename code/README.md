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

### 4. entity-tracking-lms (Kim & Schuster 2023)
- **URL**: https://github.com/sebschu/entity-tracking-lms
- **Paper**: "Entity Tracking in Language Models" (Kim & Schuster, ACL 2023)
- **Purpose**: Box-moving entity tracking benchmark to probe whether LMs can track the location of objects through sequences of move operations
- **Location**: `code/entity-tracking-lms/`

#### Key Files
- `src/dataset_generation/generate_boxes_data.py`: generates synthetic box-moving stories
- `src/evaluation/compute_metrics.py`: computes per-example and per-operation-count accuracy from model TSV output
- `scripts/data_generation/`: bash scripts reproducing all data splits from the paper (e.g., `sample_dataset_nso_exp2_max3.sh`)
- `data/boxes-dataset-v1.zip`: pre-built dataset (password: `iamnotaLM`)
- `model-outputs/model-outputs.zip`: pre-computed model predictions (password: `iamnotaLM`)

#### Dataset
Password-protected ZIP to prevent training data leakage. Unlock with password `iamnotaLM`.

#### How to Run
```bash
# Generate a dataset variant
cd entity-tracking-lms
python src/dataset_generation/generate_boxes_data.py  # see scripts/ for exact flags

# Evaluate model output (TSV with columns: target, prediction, input)
python src/evaluation/compute_metrics.py \
  --model_output <PATH>.tsv \
  --gold_data <PATH>.jsonl
```

#### Dependencies
No `requirements.txt` present; standard Python + HuggingFace Transformers assumed (check individual scripts for imports).

---

### 5. state-tracking (Li et al. 2025)
- **URL**: https://github.com/belindal/state-tracking
- **Paper**: "(How) Do Language Models Track State?" (Li, Guo & Andreas, arXiv 2503.02854, 2025)
- **Purpose**: Mechanistic study of how LMs internally track state via a synthetic permutation task (S3/S5), using probing, activation patching, and attention pattern analysis
- **Location**: `code/state-tracking/`

#### Key Files
- `permutation_task.py`: generate S3/S5 synthetic permutation stories
- `train.py`: fine-tune GPT-2/Pythia on permutation task
- `eval.py`: evaluate generalization accuracy across sequence lengths
- `bash_scripts/train.sh`, `eval.sh`, `run_intervention.sh`, `run_probe.sh`, `run_lengthwise_probe.sh`: orchestration scripts for all experiments
- `interpret/main.py`: activation patching, probing, and attention visualization
- `interpret/interpreters/`: individual mechanistic analysis modules
- `utils/models.py`, `utils/data_loaders.py`, `utils/model_utils.py`: shared utilities
- `make_topic_training_data.py`: generate synthetic topic-model pretraining data (NTP setting)

#### How to Run
```bash
# Setup
conda create -n lm_state_track python=3.12
conda activate lm_state_track
pip install -r requirements.txt
# Install PyTorch separately for your CUDA version

# Generate data (3-item permutation task)
python permutation_task.py --num_items 3 --data_dir data/S3 \
  --num_stories 10000 --train_ratio 0.9 --story_length 100

# Train
bash bash_scripts/train.sh EleutherAI/pythia-70M 3 data/S3 checkpoints/pythia_S3

# Evaluate
bash bash_scripts/eval.sh checkpoints/pythia_S3/checkpoint-XXXX 3 data/S3

# Probing
bash bash_scripts/run_probe.sh \
  --model_type pythia \
  --checkpoint_dir checkpoints/pythia_S3/checkpoint-XXXX \
  --num_items 3
```

#### Dependencies (`requirements.txt`)
numpy, tqdm, wandb, transformers==4.49.0, accelerate, matplotlib, seaborn, scikit-learn, jupyter, nnsight, plotly. PyTorch must be installed separately.

---

### 6. OpenToM
- **URL**: https://github.com/seacowx/OpenToM
- **Paper**: "OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models" (Xu et al., ACL 2024; arXiv 2402.06044)
- **Purpose**: Benchmark of 696 narratives / 16,008 questions testing LLM Neural Theory-of-Mind (N-ToM) across location, multihop, and attitude question types
- **Location**: `code/OpenToM/`

#### Key Files
- `data/opentom.json`: 13,708 QA pairs from 596 normal-length narratives
- `data/opentom_long.json`: 2,300 QA pairs from 100 long narratives
- `data/opentom_data/`: per-question-type JSONs (location_cg_fo/so, location_fg_fo/so, multihop_fo/so, attitude) plus metadata
- `src/run_baseline.py`: main experiment runner (GPT-3.5/4, Llama2, Mixtral)
- `src/evaluate.py`: macro-F1 evaluation
- `src/inference/`: model-specific inference modules (gpt, llama, mixtral, deberta_nli, etc.)
- `src/evaluate/opentom_evaluator.py`, `evaluate_plot.py`: evaluation utilities
- `src/prompts/`: prompt templates

#### How to Run
```bash
cd OpenToM/src

# Run baseline experiment (requires API keys / large GPU for open models)
python run_baseline.py \
  --model gpt4 \
  --question_type location_fg_fo \
  --lg fine

# With chain-of-thought
python run_baseline.py --model gpt35 --cot

# Evaluate saved results
python evaluate.py \
  --result_path ../data/results/<result>.json \
  --location_granularity fine \
  --perspective all
```

#### Dependencies (`src/requirements.txt`)
wandb, torch, openai, backoff, colorful, tokenizer, bert_score, scikit-learn, adapter-transformers, peft, datasets, accelerate, bitsandbytes, transformers, SentencePiece. GPU requirements: ~300 GB VRAM for Llama2-70B (full precision), ~160 GB for quantized float16.

#### Notes
- Do NOT test questions in OpenAI Playground (data contamination risk)
- Evaluation metric: macro-averaged F1 (labels are not uniformly distributed)
- HuggingFace dataset mirror: https://huggingface.co/datasets/SeacowX/OpenToM

---

## How to Use These Repositories

### For Baseline Comparisons

1. **Entity Tracking Baseline**: Use `entity-tracking-lms` to compare against the established entity tracking benchmark (box-moving task with variable operation counts)

2. **State Tracking Mechanisms**: Use `state-tracking` for mechanistic analysis (probing, activation patching) of how models internally represent tracked state across varying sequence lengths

3. **Theory-of-Mind / Mental State Tracking**: Use `OpenToM` to benchmark character belief and attitude tracking in narrative contexts

4. **Long-Context Evaluation**: Use `ruler-benchmark` to test variable tracking across different context lengths

5. **Position Effects**: Use `lost-in-the-middle` to understand if character tracking failures are position-dependent

### Recommended Workflow

```python
# Example: Use RULER's variable tracking task generation
import sys
sys.path.append('code/ruler-benchmark/synthetic')
# Import their task generation utilities

# Example: Use entity-tracking-lms evaluation metrics
sys.path.append('code/entity-tracking-lms/src')
# Import their evaluation code

# Example: Use state-tracking to probe internal representations
# See code/state-tracking/bash_scripts/ for ready-made scripts
```

---

## Integration with Our Research

Our hypothesis about character tracking limits can be tested by:

1. **Varying the number of entities** (like our synthetic dataset does; parallels the S3/S5 parameterization in `state-tracking`)
2. **Measuring where accuracy drops** (using RULER-style metrics and entity-tracking-lms evaluation)
3. **Checking for position effects** (using lost-in-the-middle analysis)
4. **Probing internal representations** (using state-tracking's probing/activation-patching pipeline)
5. **Testing belief-state tracking in narratives** (using OpenToM's location and attitude questions)

The code in these repositories provides:
- Proven evaluation frameworks
- Data generation utilities
- Mechanistic interpretability tools (probing, activation patching, attention patterns)
- Baseline implementations for comparison
