# Literature Review: How Many Characters Can a Model Keep Track Of?

## Research Area Overview

This review surveys work on language models' ability to track entities (characters, objects, variables) and their changing states across text. The core question is: **at what point does tracking break down as the number of entities or state changes increases?** This intersects entity tracking in NLP, narrative understanding, theory-of-mind benchmarks, and mechanistic interpretability of transformers.

---

## Key Papers

### 1. Entity Tracking in Language Models (Kim & Schuster, 2023)
- **Source**: ACL 2023 | arXiv:2305.02363
- **Key Contribution**: Created a controlled synthetic "Boxes" benchmark to test entity tracking without shortcuts. Showed that prior claims of entity tracking (Li et al., 2021) were inflated by trivial baselines.
- **Methodology**: 7 boxes, 100 object nouns, 12 state-changing operations per scenario. Models must predict box contents after N operations. Four harder variants test contextual references (AmbiRef, MoveContents), surface form generalization (AltForms), and compositional generalization (NumOps).
- **Datasets Used**: New "Boxes" dataset (7 variants); reanalysis of Alchemy and TextWorld
- **Key Results**:
  - GPT-3 and Flan-T5: near-zero accuracy on non-trivial tracking (essentially repeat initial state)
  - GPT-3.5 (code-pretrained): >25% accuracy even after 7 operations per box, well above random baseline
  - **Code pretraining is the key factor**, not RLHF or instruction tuning
  - Finetuned T5-base achieves near-perfect accuracy on base split, proving the task is learnable
  - Performance degrades monotonically with operation count; steepest drop at 0-3 operations
  - Context-dependent operations (MoveContents, AmbiRef) accelerate breakdown
- **Code**: https://github.com/sebschu/entity-tracking-lms
- **Relevance**: **Central paper**. Directly measures how many state changes models can track per entity across 7 simultaneous entities.

### 2. (How) Do Language Models Track State? (Li, Guo & Andreas, 2025)
- **Source**: ICML 2025 | arXiv:2503.02854
- **Key Contribution**: Mechanistic interpretability study revealing two distinct algorithms transformers learn for state tracking: Associative Algorithm (AA) and Parity-Associative Algorithm (PAA).
- **Methodology**: Permutation composition task (S3 and S5 symmetric groups) as a principled proxy for state tracking. S5 is NC1-complete, meaning any finite-state tracking task can be reduced to it. Uses probing, activation patching, and attention analysis on Pythia-160M and GPT-2 family models.
- **Datasets Used**: 1M synthetic permutation sequences of length 100
- **Key Results**:
  - Models learn one of two algorithms: AA (hierarchical parallel scan, log-depth) or PAA (parity-first parallel computation, then associative scan for remainder)
  - AA generalizes better to longer sequences; PAA is more brittle
  - Architecture and initialization determine which algorithm emerges, **not model size**
  - Models generalize perfectly up to training length (100), then degrade sharply at a "cutoff length"
  - Code/topic-model pretraining steers toward AA; parity pretraining steers toward PAA
  - Same associative signatures appear in natural-language versions of the task
- **Code**: https://github.com/belindal/state-tracking
- **Relevance**: Reveals the **internal mechanisms** for state tracking — important for understanding *why* models fail at higher entity counts.

### 3. Code Pretraining Improves Entity Tracking (Kim, Schuster & Toshniwal, 2024)
- **Source**: arXiv:2405.21068
- **Key Contribution**: Systematic confirmation that code pretraining causally improves entity tracking, using matched model pairs (base vs. code, holding architecture constant).
- **Methodology**: Same Boxes task. Compared Llama 2 vs Code Llama, DeepSeek vs DeepSeek-Coder, Gemma vs CodeGemma across 7B/13B/70B scales. Also tested math-trained and alignment-tuned variants.
- **Key Results**:
  - Code training consistently helps; amount of code data matters (2T > 500B tokens)
  - Math training provides minimal benefit
  - **Best result: Code Llama 70B-Instruct at 64.9% on non-trivial examples**
  - 7B models cannot reliably beat random baseline at 5-7 operations — hard limit for small models
  - Alignment tuning helps base models more than code models
- **Relevance**: Quantifies the **scaling limits** — even the best 70B code model only achieves 65% on multi-step tracking.

### 4. Too Long, Didn't Model (TLDM) (Hamilton et al., 2025)
- **Source**: arXiv:2505.14925
- **Key Contribution**: Benchmark showing all frontier LLMs lose stable narrative understanding beyond 64K tokens, with entity/character tracking (storyworld description) degrading fastest of three tasks.
- **Methodology**: 40 English novels from Project Gutenberg. Three tasks: summarization, storyworld description (character locations), narrative time estimation. Compares full-novel outputs to chapter-level (short-context) outputs across 7 frontier models (GPT-4.1, Llama 4 Scout, DeepSeek V3, Gemini 2.0 Flash, Gemma 3, Qwen 3, Mistral Small).
- **Key Results**:
  - **Storyworld tracking (character locations) collapses most sharply** with novel length
  - No model retains stable understanding beyond **64K tokens** despite claiming 1M+ context windows
  - GPT-4.1 most robust; open-weight models degrade fastest
  - Truncating to relevant section consistently improves performance — extraneous context hurts
  - Performance scales linearly with parameter count
  - Shuffling chapters hurts storyworlds but not time estimates
- **Relevance**: **Directly measures character tracking in real narratives** at scale.

### 5. CHATTER: Character Attribution for Narrative Understanding (Baruah & Narayanan, 2025)
- **Source**: arXiv:2411.05227
- **Key Contribution**: Largest character attribution dataset (88K character-trope pairs from 660 movies, 2,998 characters, 12,967 tropes). Tests whether LLMs can identify character traits from screenplays.
- **Methodology**: Binary classification of character-trope pairs. Full screenplay context (~42K tokens average, up to 158K). Five prompting strategies across closed and open-source models.
- **Key Results**:
  - Full script context actually *decreases* performance for most models (vs. priors alone)
  - Gemini-1.5-Flash achieves 81.9% accuracy with priors; drops to 72.4% with full script
  - Zero-shot segment prompting outperforms few-shot — adding examples hurts
  - Models likely exploit pretraining knowledge of TVTropes
- **Relevance**: Shows models fail to aggregate **distributed character information** across long contexts.

### 6. OpenToM: Theory-of-Mind Benchmark (Xu et al., 2024)
- **Source**: ACL 2024 | arXiv:2402.06044
- **Key Contribution**: ToM benchmark with 696 narratives, 16K questions testing physical-world and psychological mental state tracking for multiple characters.
- **Code**: https://github.com/seacowx/OpenToM
- **Relevance**: Tests per-character belief and location tracking — core to the research question.

### 7. Locations of Characters in Narratives (2025)
- **Key Contribution**: Manually annotated datasets for character location tracking in fairy tales (Andersen, 15 stories) and novels (Persuasion).
- **Key Results**: Best LLM achieves only ~62% on Andersen, ~56% on Persuasion.
- **Relevance**: Direct benchmark for spatial entity tracking with low accuracy.

### 8. Finding Flawed Fictions (2025)
- **Key Contribution**: FLAWEDFICTIONS benchmark for plot hole detection. LLM performance degrades sharply as story length increases.
- **Relevance**: Plot hole detection requires tracking character states and story consistency.

### 9. Mary, the Cheeseburger-Eating Vegetarian (2025)
- **Key Contribution**: Shows LLMs' internal representations can detect character incoherence, but generated responses cannot reliably distinguish coherent from incoherent narratives. More sensitive to world-knowledge violations than character-trait violations.
- **Relevance**: Demonstrates gap between representation and behavior for character tracking.

### 10. EvolvTrip: Temporal Theory-of-Mind Graphs (2025)
- **Key Contribution**: Proposes external knowledge graphs to compensate for LLM failures in tracking evolving character mental states in literature. Creates LitCharToM benchmark.
- **Relevance**: Addresses the failure mode and proposes augmentation strategy.

### 11. SCORE: Story Coherence via RAG (2025)
- **Key Contribution**: Uses item status tracking + episode summaries + RAG for narrative coherence.
- **Relevance**: Engineering solution implies the character tracking limitation exists and needs external augmentation.

### 12. Tracking Discrete and Continuous Entity State (Dalvi et al., 2019)
- **Source**: NAACL 2019 | arXiv:1904.03518
- **Key Contribution**: Foundational structured prediction for entity state tracking in procedural text (ProPara dataset).
- **Relevance**: Established task formulation and evaluation metrics.

### 13. Lost in the Middle (Liu et al., 2024)
- **Source**: TACL 2024 | arXiv:2307.03172
- **Key Contribution**: U-shaped performance curve — models use information at beginning and end of context better than the middle.
- **Relevance**: Characters introduced mid-story may be tracked worse.

### 14. RULER Benchmark (Hsieh et al., 2024)
- **Source**: COLM 2024 | arXiv:2404.06654
- **Key Contribution**: Comprehensive long-context benchmark including Variable Tracking task (multi-hop variable assignment chains).
- **Relevance**: Variable tracking is directly analogous to character attribute tracking.

---

## Common Methodologies

1. **Synthetic controlled tasks**: Boxes task (Kim & Schuster), permutation composition (Li et al.), RULER variable tracking — precisely control entity count and operation count
2. **Benchmark evaluation on narratives**: TLDM, CHATTER, OpenToM, NoCha — test on real/semi-real stories
3. **Probing and mechanistic analysis**: Linear probes on hidden states, activation patching, attention pattern analysis (Li et al.)
4. **Scaling analysis**: Vary model size, context length, number of entities/operations to find breakdown points

## Standard Baselines

- **Random baseline**: Sample from contextually mentioned entities (Kim & Schuster)
- **Initial state copying**: Predict no change from initial state (trivially strong when most states unchanged)
- **Chapter-level performance**: Compare full-novel to per-chapter results (TLDM)
- **Priors-only**: Use model's pretraining knowledge without context (CHATTER)

## Evaluation Metrics

- **Exact match accuracy** by number of operations (Boxes task)
- **Semantic similarity** between full-context and short-context outputs (TLDM)
- **Jaccard similarity** for entity set comparison (TLDM storyworlds)
- **F1 score** for character attribute classification (CHATTER, OpenToM)
- **Probe accuracy** at each layer for mechanistic analysis

## Datasets in the Literature

| Dataset | Task | Entities | Scale | Used In |
|---------|------|----------|-------|---------|
| Boxes | Object tracking across moves | 7 boxes, ~20 objects | 90K examples | Kim & Schuster 2023, 2024 |
| Permutation sequences | State composition | 3-5 elements | 1M sequences | Li et al. 2025 |
| OpenToM | Character belief tracking | 2-3 characters | 16K questions | Xu et al. 2024 |
| TLDM (40 novels) | Character location tracking | Full cast | 40 novels | Hamilton et al. 2025 |
| CHATTER | Character-trope attribution | 2,998 characters | 88K pairs | Baruah & Narayanan 2025 |
| ProPara | Process entity states | 2-5 entities | 445 paragraphs | Dalvi et al. 2019 |
| NarrativeQA | Narrative comprehension | Variable | 33K QA pairs | Kočiský et al. 2018 |
| SimpleToM | Theory of mind | 2 characters | 1,147 stories | allenai |
| Andersen/Persuasion | Character locations | ~5-15 per story | 16 texts | 2025 |
| RULER VT | Variable tracking | 2-10 chains | Configurable | Hsieh et al. 2024 |
| NoCha | Claim verification | Novel characters | 1,001 pairs | Karpinska et al. 2024 |
| FLAWEDFICTIONS | Plot hole detection | Variable | Configurable | 2025 |

---

## Gaps and Opportunities

1. **No systematic character count scaling study**: No paper systematically varies the number of characters (e.g., 2, 4, 8, 16, 32) while holding story length constant. Kim & Schuster fix at 7 boxes; RULER uses fixed chain patterns.
2. **Synthetic vs. natural gap**: Boxes task is controlled but unrealistic; novel benchmarks are realistic but uncontrolled. A middle ground (synthetic stories with controllable character count) is needed.
3. **Character attributes vs. character count**: Most work tracks one attribute per entity. How many distinct attributes per character can be tracked simultaneously?
4. **Modern model evaluation**: Kim & Schuster (2023) tested GPT-3.5; need evaluation of Claude, GPT-4, Gemini, Llama 3, etc.
5. **Confusion vs. forgetting**: When models fail, do they confuse characters (swap attributes) or lose information entirely? This distinction has experimental implications.
6. **Interaction between character count and context length**: More characters means longer text — these effects need disentangling.

---

## Recommendations for Our Experiment

### Recommended Approach
Design a controlled experiment that systematically varies the number of characters in a story while testing the model's ability to track per-character attributes. Use synthetic story generation to control:
- Number of characters (2, 4, 8, 16, 32, 64)
- Number of attributes per character (name, location, possession, relationship)
- Number of state changes per character
- Total story length

### Recommended Datasets
1. **Primary**: Generate synthetic stories extending the Boxes paradigm to character narratives (use entity-tracking-lms code as starting point)
2. **Validation**: OpenToM (multi-character belief tracking) and ProPara (entity state tracking)
3. **Real-world validation**: TLDM-style evaluation on novels (optional, for ecological validity)

### Recommended Baselines
- Random guessing from mentioned entities
- Initial state copying (predict no change)
- Most-recent-mention heuristic
- Compare across model families and sizes (including code-pretrained variants)

### Recommended Metrics
- Exact match accuracy by character count and operation count
- Per-character accuracy (does the model confuse Character A's attributes with Character B's?)
- Confusion matrix between characters (are nearby characters more confused?)
- Degradation curve: accuracy as a function of character count

### Methodological Considerations
- **Control for context length**: More characters = longer text — disentangle character count from length effects
- **Code-pretrained models**: Literature strongly suggests these will perform best
- **Attribute type matters**: Location tracking (concrete) may be easier than belief/intention tracking (abstract)
- **Synthetic stories should be natural-sounding**: Use templates that read like actual narratives, not procedural instructions
- **Position effects**: Characters introduced mid-story may be tracked worse (Lost in the Middle)
- **Test both state-changed and unchanged entities**: Many benchmarks are inflated by trivial "no change" cases (Kim & Schuster's key insight)
