# Resources Catalog

This document catalogs all resources gathered for the research project on character tracking limits in language models.

## Summary

| Resource Type | Count | Location |
|---------------|-------|----------|
| Papers | 7 | `papers/` |
| Datasets | 3 | `datasets/` |
| Code Repositories | 3 | `code/` |

---

## Papers

Total papers downloaded: **7**

| Title | Authors | Year | File | Key Contribution |
|-------|---------|------|------|-----------------|
| Entity Tracking in Language Models | Kim & Schuster | 2023 | `papers/2305.02363_entity_tracking_in_language_models.pdf` | First systematic entity tracking evaluation |
| Lost in the Middle | Liu et al. | 2024 | `papers/2307.03172_lost_in_the_middle.pdf` | U-shaped context usage curve |
| RULER Benchmark | Hsieh et al. | 2024 | `papers/2404.06654_ruler_benchmark.pdf` | Variable tracking task |
| NoCha Benchmark | Karpinska et al. | 2024 | `papers/2406.16264_nocha_benchmark.pdf` | Book-length comprehension |
| How Do LMs Track State? | Li et al. | 2025 | `papers/2503.02854_how_do_lms_track_state.pdf` | Mechanistic analysis |
| Too Long, Didn't Model | Various | 2025 | `papers/2505.14925_too_long_didnt_model.pdf` | Context failure analysis |
| CharacterBench | Zhou et al. | 2024 | `papers/2412.11912_character_bench.pdf` | Character customization evaluation |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: **3**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Character Tracking Synthetic | Generated | 90 examples | State tracking | `datasets/character_tracking_synthetic.json` | Primary experiment dataset |
| bAbI Tasks | Facebook | 10K per task | QA/Reasoning | `datasets/tasks_1-20_v1-2/` | Entity tracking baseline |
| NarrativeQA Samples | DeepMind | 10 samples | Story QA | `datasets/narrativeqa_samples.json` | Reference examples |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: **3**

| Name | URL | Purpose | Location | Key Files |
|------|-----|---------|----------|-----------|
| entity-tracking-lms | github.com/sebschu/entity-tracking-lms | Entity tracking experiments | `code/entity-tracking-lms/` | `src/`, `scripts/` |
| RULER | github.com/NVIDIA/RULER | Long-context benchmarking | `code/ruler-benchmark/` | `synthetic/`, `scripts/` |
| lost-in-the-middle | github.com/nelson-liu/lost-in-the-middle | Position-based analysis | `code/lost-in-the-middle/` | `src/`, `scripts/` |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Paper Search**:
   - Primary sources: arXiv, Semantic Scholar, Papers with Code
   - Keywords: "entity tracking", "character tracking", "LLM", "long context", "state tracking"
   - Focus: 2023-2025 papers for state-of-the-art

2. **Dataset Search**:
   - Checked HuggingFace Datasets for NarrativeQA
   - Downloaded bAbI from official Facebook source
   - Created custom synthetic dataset for controlled experiments

3. **Code Search**:
   - Identified repositories from paper links
   - Cloned with `--depth 1` to minimize size
   - Focused on repositories with evaluation code

### Selection Criteria

**Papers selected based on**:
- Direct relevance to entity/character tracking
- Methodology applicable to our research
- Established benchmarks for comparison
- Mechanistic insights into LLM capabilities

**Datasets selected based on**:
- Controllable entity/character counts
- Clear ground truth for evaluation
- Synthetic generation capability
- Established baselines available

**Code selected based on**:
- Official implementations from papers
- Reusable evaluation frameworks
- Data generation utilities

### Challenges Encountered

1. **bAbI loading**: HuggingFace dataset loader deprecated; used direct download instead
2. **Large datasets**: NarrativeQA full texts too large; used samples only
3. **Paper access**: All target papers available on arXiv (no paywalls)

### Gaps and Workarounds

| Gap | Workaround |
|-----|------------|
| No existing character-count-varying dataset | Created synthetic generator |
| bAbI uses fixed entity counts | Use for baseline comparison only |
| NoCha uses copyrighted novels | Use for methodology reference only |

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset(s)
**Use `character_tracking_synthetic.json`** as the primary evaluation dataset because:
- Systematically varies character count (2-20)
- Includes multiple attribute types (location, mood, holding)
- Allows controlled complexity experiments
- Ground truth is programmatically generated

### 2. Baseline Methods
Compare against:
1. **Random baseline**: Predict most common answer
2. **First-state baseline**: Predict initial state (ignore changes)
3. **bAbI Task 2 performance**: Established benchmark
4. **RULER Variable Tracking**: Multi-hop baseline

### 3. Evaluation Metrics
Use metrics from the literature:
- **Overall accuracy** (standard)
- **Per-character-count accuracy** (our primary analysis)
- **Per-question-type accuracy** (location/mood/holding)
- **Position-based accuracy** (early vs. late characters)

### 4. Code to Adapt/Reuse
- **From entity-tracking-lms**: Evaluation scripts, data generation patterns
- **From RULER**: Task generation framework, multi-hop tracking patterns
- **From lost-in-the-middle**: Position analysis methodology

---

## Experiment Runner Instructions

The experiment runner should:

1. **Load the synthetic dataset**:
```python
import json
with open("datasets/character_tracking_synthetic.json") as f:
    data = json.load(f)
```

2. **Iterate through configurations**:
- Group examples by `num_characters` and `num_actions`
- Compute accuracy for each configuration
- Plot accuracy vs. character count

3. **Use bAbI as sanity check**:
- Parse `datasets/tasks_1-20_v1-2/en/qa2_two-supporting-facts_test.txt`
- Verify model can do basic entity tracking before complex experiments

4. **Reference code repositories for**:
- Prompt templates (see entity-tracking-lms)
- Evaluation metrics (see RULER)
- Position analysis (see lost-in-the-middle)

---

## File Manifest

```
project/
├── papers/
│   ├── 2305.02363_entity_tracking_in_language_models.pdf
│   ├── 2307.03172_lost_in_the_middle.pdf
│   ├── 2404.06654_ruler_benchmark.pdf
│   ├── 2406.16264_nocha_benchmark.pdf
│   ├── 2412.11912_character_bench.pdf
│   ├── 2503.02854_how_do_lms_track_state.pdf
│   └── 2505.14925_too_long_didnt_model.pdf
├── datasets/
│   ├── .gitignore
│   ├── README.md
│   ├── character_tracking_synthetic.json
│   ├── narrativeqa_samples.json
│   └── tasks_1-20_v1-2/
│       └── en/
│           ├── qa1_single-supporting-fact_*.txt
│           ├── qa2_two-supporting-facts_*.txt
│           └── ... (20 tasks total)
├── code/
│   ├── README.md
│   ├── entity-tracking-lms/
│   ├── ruler-benchmark/
│   └── lost-in-the-middle/
├── literature_review.md
├── resources.md
└── .resource_finder_complete
```
