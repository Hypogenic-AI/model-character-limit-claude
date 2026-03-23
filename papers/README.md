# Downloaded Papers

This directory contains research papers relevant to the character tracking limits study.

## Core Entity Tracking Papers

### 1. Entity Tracking in Language Models
- **File**: `2305.02363_Entity_Tracking_in_Language_Models.pdf`
- **Authors**: Najoung Kim, Sebastian Schuster
- **Year**: 2023 | **Venue**: ACL 2023
- **Why relevant**: Central paper. Boxes benchmark (7 boxes, 12 ops). Code pretraining is key factor. Code: github.com/sebschu/entity-tracking-lms

### 2. Code Pretraining Improves Entity Tracking
- **File**: `2405.21068_Code_Pretraining_Improves_Entity_Tracking.pdf`
- **Authors**: Kim, Schuster, Toshniwal
- **Year**: 2024
- **Why relevant**: Systematic code vs math vs alignment comparison. Best: Code Llama 70B-Instruct at 64.9%.

### 3. (How) Do Language Models Track State?
- **File**: `2503.02854_How_Do_Language_Models_Track_State.pdf`
- **Authors**: Li, Guo, Andreas
- **Year**: 2025 | **Venue**: ICML 2025
- **Why relevant**: Mechanistic analysis — AA vs PAA algorithms. Code: github.com/belindal/state-tracking

### 4. Efficient and Interpretable Entity Tracking
- **File**: `2208.14252_Efficient_Interpretable_Entity_Tracking.pdf`
- **Year**: 2022
- **Why relevant**: PhD dissertation on neural entity tracking architectures.

### 5. Effective Use of Transformers for Entity Tracking
- **File**: `1909.02635_Effective_Use_Transformers_Entity_Tracking.pdf`
- **Year**: 2019
- **Why relevant**: Early work on transformer entity tracking.

### 6. Tracking Discrete and Continuous Entity State
- **File**: `1904.03518_Tracking_Discrete_Continuous_Entity_State.pdf`
- **Authors**: Dalvi et al.
- **Year**: 2019 | **Venue**: NAACL 2019
- **Why relevant**: ProPara dataset, foundational structured prediction for entity states.

## Character/Narrative Understanding Papers

### 7. Too Long, Didn't Model (TLDM)
- **File**: `2025_Too_Long__Didn_t_Model__Decomposing_LLM_Long_Context_Underst.pdf`
- **Authors**: Hamilton, Hicke, Wilkens, Mimno
- **Year**: 2025
- **Why relevant**: 40 novels. Character location tracking collapses >64K tokens. All 7 frontier models tested.

### 8. CHATTER: Character Attribution for Narrative Understanding
- **File**: `2411.05227_CHATTER_Character_Attribution_Narrative.pdf`
- **Authors**: Baruah & Narayanan
- **Year**: 2025
- **Why relevant**: 88K character-trope pairs from 660 movies. Full script context hurts performance.

### 9. OpenToM
- **File**: `2024_OpenToM__A_Comprehensive_Benchmark_for_Evaluating_Theory_of_.pdf`
- **Authors**: Xu et al.
- **Year**: 2024 | **Venue**: ACL 2024
- **Why relevant**: 16K ToM questions, multi-character belief tracking. Code: github.com/seacowx/OpenToM

### 10. Finding Flawed Fictions
- **File**: `2025_Finding_Flawed_Fictions__Evaluating_Complex_Reasoning_in_Lan.pdf`
- **Year**: 2025
- **Why relevant**: Plot hole detection benchmark. Performance degrades with story length.

### 11. Mary, the Cheeseburger-Eating Vegetarian
- **File**: `2025_Mary__the_Cheeseburger_Eating_Vegetarian__Do_LLMs_Recognize_.pdf`
- **Year**: 2025
- **Why relevant**: LLM internal representations detect character incoherence but responses can't.

### 12. EvolvTrip
- **File**: `2025_EvolvTrip__Enhancing_Literary_Character_Understanding_with_T.pdf`
- **Year**: 2025
- **Why relevant**: Temporal ToM graphs for character understanding in literature.

### 13. Locations of Characters in Narratives
- **File**: `2025_Locations_of_Characters_in_Narratives__Andersen_and_Persuasi.pdf`
- **Year**: 2025
- **Why relevant**: Annotated datasets. Best LLM: ~62% Andersen, ~56% Persuasion.

### 14. SCORE: Story Coherence via RAG
- **File**: `2025_SCORE__Story_Coherence_and_Retrieval_Enhancement_for_AI_Narr.pdf`
- **Year**: 2025
- **Why relevant**: RAG-based framework for narrative coherence via entity tracking.

### 15. CREFT: Character Relation Extraction
- **File**: `2025_CREFT__Sequential_Multi_Agent_LLM_for_Character_Relation_Ext.pdf`
- **Year**: 2025
- **Why relevant**: Multi-agent LLM for character relation extraction.

### 16. Modeling Naive Psychology of Characters
- **File**: `2018_Modeling_Naive_Psychology_of_Characters_in_Simple_Commonsens.pdf`
- **Year**: 2018
- **Why relevant**: Character mental state modeling in commonsense stories.

## Long-Context and Benchmarking Papers

### 17. Lost in the Middle
- **File**: `2307.03172_lost_in_the_middle.pdf`
- **Authors**: Liu et al.
- **Year**: 2024 | **Venue**: TACL 2024
- **Why relevant**: U-shaped context usage curve. Position effects matter for character tracking.

### 18. RULER Benchmark
- **File**: `2404.06654_ruler_benchmark.pdf`
- **Authors**: Hsieh et al.
- **Year**: 2024 | **Venue**: COLM 2024
- **Why relevant**: Variable tracking task analogous to character attributes.

### 19. NoCha Benchmark
- **File**: `2406.16264_nocha_benchmark.pdf`
- **Authors**: Karpinska et al.
- **Year**: 2024 | **Venue**: EMNLP 2024
- **Why relevant**: Book-length claim verification. Best model only 55.8%.

### 20. CharacterBench
- **File**: `2412.11912_character_bench.pdf`
- **Authors**: Zhou et al.
- **Year**: 2024 | **Venue**: AAAI 2025
- **Why relevant**: Character customization evaluation, 22K samples, 3,956 characters.

## Additional Papers

### 21-28. Other downloaded papers
- MET-Bench (multimodal entity tracking, 2025)
- Temporal Embeddings for Narrative Understanding (2020)
- StoryReasoning Dataset (2025)
- SeriesBench (drama series understanding, 2025)
- Story Ribbons (storyline visualization, 2025)
- If an LLM Were a Character (lifelong learning, 2025)
- StoryTeller (video character identification, 2024)
- SkyScript-100M (drama scripts, 2024)
- Movie Facts and Fibs MF2 (long movie understanding, 2025)

## Chunked PDFs

The `pages/` subdirectory contains chunked PDF files for deep reading. Generated using `pdf_chunker.py` with 3 pages per chunk. Key papers chunked:
- Entity Tracking in Language Models (7 chunks)
- How Do Language Models Track State (8 chunks)
- Code Pretraining Improves Entity Tracking (6 chunks)
- CHATTER (4 chunks)
- Too Long, Didn't Model (4 chunks)

## Citation Information

See `literature_review.md` for full synthesis of findings.
