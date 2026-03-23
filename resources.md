# Resources Catalog

This document catalogs all resources gathered for the research project on character tracking limits in language models.

## Summary

| Resource Type | Count | Location |
|---------------|-------|----------|
| Papers | 28+ | `papers/` |
| Datasets | 8 | `datasets/` |
| Code Repositories | 5 | `code/` |

---

## Papers

Total unique papers downloaded: **~28** (some duplicates from prior runs)

### Core Entity Tracking Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Entity Tracking in Language Models | Kim & Schuster | 2023 | `papers/2305.02363_Entity_Tracking_in_Language_Models.pdf` | Boxes benchmark, 7 entities, code pretraining key |
| Code Pretraining Improves Entity Tracking | Kim, Schuster, Toshniwal | 2024 | `papers/2405.21068_Code_Pretraining_Improves_Entity_Tracking.pdf` | Systematic code vs math vs alignment comparison |
| (How) Do Language Models Track State? | Li, Guo, Andreas | 2025 | `papers/2503.02854_How_Do_Language_Models_Track_State.pdf` | Mechanistic analysis: AA vs PAA algorithms |
| Efficient and Interpretable Entity Tracking | — | 2022 | `papers/2208.14252_Efficient_Interpretable_Entity_Tracking.pdf` | PhD dissertation on neural entity tracking |
| Effective Use of Transformers for Entity Tracking | — | 2019 | `papers/1909.02635_Effective_Use_Transformers_Entity_Tracking.pdf` | Early transformer entity tracking |
| Tracking Discrete and Continuous Entity State | Dalvi et al. | 2019 | `papers/1904.03518_Tracking_Discrete_Continuous_Entity_State.pdf` | ProPara, structured prediction for process understanding |

### Character/Narrative Understanding Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Too Long, Didn't Model (TLDM) | Hamilton et al. | 2025 | `papers/2025_Too_Long__Didn_t_Model__*.pdf` | 40 novels, character tracking collapses >64K tokens |
| CHATTER | Baruah & Narayanan | 2025 | `papers/2411.05227_CHATTER_Character_Attribution_Narrative.pdf` | 88K character-trope pairs from 660 movies |
| OpenToM | Xu et al. | 2024 | `papers/2024_OpenToM__*.pdf` | 16K ToM questions, multi-character belief tracking |
| Finding Flawed Fictions | — | 2025 | `papers/2025_Finding_Flawed_Fictions__*.pdf` | Plot hole detection, degrades with story length |
| Mary, the Cheeseburger-Eating Vegetarian | — | 2025 | `papers/2025_Mary__the_Cheeseburger_Eating_Vegetarian__*.pdf` | LLM character incoherence detection |
| EvolvTrip | — | 2025 | `papers/2025_EvolvTrip__*.pdf` | Temporal ToM graphs for character understanding |
| Locations of Characters in Narratives | — | 2025 | `papers/2025_Locations_of_Characters_in_Narratives__*.pdf` | Andersen & Persuasion character location datasets |
| SCORE | — | 2025 | `papers/2025_SCORE__*.pdf` | RAG for narrative coherence |
| CREFT | — | 2025 | `papers/2025_CREFT__*.pdf` | Multi-agent LLM for character relation extraction |
| Modeling Naive Psychology of Characters | — | 2018 | `papers/2018_Modeling_Naive_Psychology__*.pdf` | Character mental state modeling |

### Long-Context and Benchmarking Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Lost in the Middle | Liu et al. | 2024 | `papers/2307.03172_lost_in_the_middle.pdf` | U-shaped context usage |
| RULER Benchmark | Hsieh et al. | 2024 | `papers/2404.06654_ruler_benchmark.pdf` | Variable tracking task |
| NoCha Benchmark | Karpinska et al. | 2024 | `papers/2406.16264_nocha_benchmark.pdf` | Book-length claim verification |
| CharacterBench | Zhou et al. | 2024 | `papers/2412.11912_character_bench.pdf` | Character customization evaluation |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: **8**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Boxes v1 | Kim & Schuster 2023 | 90K test examples | Entity state tracking | `datasets/boxes/` | 7 boxes, 12 operations, primary baseline |
| OpenToM | Xu et al. 2024 | 13,708 QA examples | Character belief/location tracking | `datasets/opentom/` | Multi-character ToM |
| ProPara | Allen AI | 445 paragraphs | Process entity states | `datasets/propara/` | Procedural text entity tracking |
| SimpleToM | Allen AI | 1,147 stories | Theory of mind | `datasets/simpletom/` | 2-character ToM |
| NarrativeQA | DeepMind | 33K QA pairs | Narrative comprehension | `datasets/narrativeqa/` | Full novels/scripts |
| Theory of Mind | grimulkan | 539 examples | ToM instruction following | `datasets/theory_of_mind_grimulkan/` | Small ToM dataset |
| Synthetic Character Tracking | Generated | 90 examples | Character state tracking | `datasets/character_tracking_synthetic.json` | From prior run |
| Samples | Various | Small | Reference | `datasets/samples/` | Example records from each dataset |

See `datasets/README.md` for detailed download instructions and loading code.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Key Files |
|------|-----|---------|----------|-----------|
| entity-tracking-lms | github.com/sebschu/entity-tracking-lms | Boxes task data generation + evaluation | `code/entity-tracking-lms/` | `src/dataset_generation/generate_boxes_data.py`, `src/evaluation/` |
| state-tracking | github.com/belindal/state-tracking | Mechanistic analysis of state tracking | `code/state-tracking/` | `permutation_task.py`, `train.py`, `interpret/` |
| OpenToM | github.com/seacowx/OpenToM | ToM benchmark evaluation | `code/OpenToM/` | `data/opentom.json`, `src/run_baseline.py` |
| ruler-benchmark | github.com/NVIDIA/RULER | Long-context variable tracking | `code/ruler-benchmark/` | `synthetic/`, `scripts/` |
| lost-in-the-middle | github.com/nelson-liu/lost-in-the-middle | Position-based analysis | `code/lost-in-the-middle/` | `src/`, `scripts/` |

See `code/README.md` for detailed descriptions and usage instructions.

---

## Resource Gathering Notes

### Search Strategy

1. **Paper Search**: Used paper-finder service (diligent mode, 150 results), arxiv API (3 targeted queries), and Semantic Scholar API. Keywords: "entity tracking language models", "character tracking narrative LLM", "state tracking transformer scalability".
2. **Dataset Search**: HuggingFace Datasets API, direct downloads from paper repositories, synthetic generation from entity-tracking-lms code.
3. **Code Search**: GitHub repos linked in papers, GitHub search for "OpenToM".

### Selection Criteria

Papers prioritized by: (1) direct relevance to entity/character count limits, (2) controlled experimental methodology, (3) mechanistic insights, (4) established benchmarks.

Datasets prioritized by: (1) controllable entity count, (2) clear ground truth, (3) availability and ease of use.

### Challenges Encountered

1. **Semantic Scholar rate limiting**: API returned 429 for ~7 papers; resolved with 3-5s delays between requests. 2 papers still could not be downloaded.
2. **CHATTER dataset**: Not publicly released; GitHub repo contains only crawling scripts, not pre-built data.
3. **Duplicate papers**: Some papers downloaded in both prior and current runs with slightly different filenames.

### Gaps and Workarounds

| Gap | Workaround |
|-----|------------|
| No dataset systematically varying character count | Recommend generating synthetic stories extending Boxes paradigm |
| CHATTER dataset unavailable | Use CHATTER methodology description for experiment design |
| TLDM novel data not packaged | Use Project Gutenberg novels directly |

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Generate synthetic character tracking stories** with controllable parameters: number of characters (2, 4, 8, 16, 32, 64), attributes per character, state changes per character
- Use `code/entity-tracking-lms/` as starting point for data generation
- Validate with **OpenToM** and **ProPara** as established benchmarks

### 2. Baseline Methods
1. Random guessing from mentioned entities
2. Initial state copying (predict no change)
3. Most-recent-mention heuristic
4. Compare code-pretrained vs base models (literature strongly predicts code models will win)

### 3. Evaluation Metrics
- **Exact match accuracy** by character count (primary)
- **Per-character accuracy** (confusion analysis)
- **Degradation curve**: accuracy as function of character count
- **Confusion matrix**: which characters get mixed up

### 4. Code to Adapt/Reuse
- **entity-tracking-lms**: Data generation scripts, evaluation framework
- **OpenToM**: Multi-character evaluation pipeline
- **RULER**: Variable tracking task generation (analogous to character attributes)

### 5. Key Methodological Insights from Literature
- **Separate trivial from non-trivial cases** (Kim & Schuster's key insight — many benchmarks are inflated)
- **Control for context length** when varying character count
- **Test code-pretrained models** — they will likely outperform
- **Characters mid-story may be tracked worse** (Lost in the Middle effect)
- **Storyworld/location tracking degrades faster** than other narrative tasks (TLDM finding)

---

## File Manifest

```
project/
├── papers/                          # ~28 PDFs
│   ├── README.md
│   ├── pages/                       # Chunked PDFs for reading
│   ├── 2305.02363_Entity_Tracking_in_Language_Models.pdf
│   ├── 2503.02854_How_Do_Language_Models_Track_State.pdf
│   ├── 2405.21068_Code_Pretraining_Improves_Entity_Tracking.pdf
│   ├── 2411.05227_CHATTER_Character_Attribution_Narrative.pdf
│   ├── 2025_Too_Long__Didn_t_Model__*.pdf
│   └── ... (20+ more)
├── datasets/
│   ├── .gitignore
│   ├── README.md
│   ├── samples/                     # Small samples from each dataset
│   ├── boxes/                       # Kim & Schuster Boxes task (90K examples)
│   ├── opentom/                     # OpenToM ToM benchmark (13K QA)
│   ├── propara/                     # ProPara entity states (445 paragraphs)
│   ├── simpletom/                   # SimpleToM (1,147 stories)
│   ├── narrativeqa/                 # NarrativeQA (33K QA pairs)
│   └── theory_of_mind_grimulkan/    # ToM examples (539)
├── code/
│   ├── README.md
│   ├── entity-tracking-lms/         # Boxes task generation + eval
│   ├── state-tracking/              # Mechanistic analysis
│   ├── OpenToM/                     # ToM benchmark
│   ├── ruler-benchmark/             # RULER long-context benchmark
│   └── lost-in-the-middle/          # Position analysis
├── literature_review.md             # Comprehensive lit review
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker
```
