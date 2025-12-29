# Literature Review: Character Tracking Limits in Language Models

## Research Area Overview

This literature review examines the ability of large language models (LLMs) to track entities, characters, and their changing states across textual narratives. The research question focuses on **how many characters a model can simultaneously track** and what mechanisms underlie this capability.

The field spans several related areas:
1. **Entity/State Tracking**: Tracking how objects and entities change as text unfolds
2. **Long-Context Understanding**: How models use information across extended inputs
3. **Working Memory in LLMs**: Analogies to human cognitive limits
4. **Narrative Comprehension**: Understanding character roles, attributes, and relationships

---

## Key Papers

### Paper 1: Entity Tracking in Language Models
- **Authors**: Najoung Kim, Sebastian Schuster
- **Year**: 2023
- **Source**: ACL 2023
- **File**: `papers/2305.02363_entity_tracking_in_language_models.pdf`

#### Key Contribution
First systematic investigation of whether LLMs can track entities through state changes in discourse. Proposes a controlled evaluation using a box-moving task where objects are placed in boxes and moved between them.

#### Methodology
- **Task**: Given initial state of boxes containing objects + sequence of move operations, predict final state
- **Example**: "Box 1 contains the book. Move the book to Box 2. Box 2 contains ____"
- **Dataset**: Synthetically generated with 7 boxes, up to 12 operations
- **Models tested**: GPT-3, GPT-3.5, Flan-T5

#### Key Findings
1. **Only GPT-3.5 (code-trained) shows entity tracking ability** - text-only pretrained models fail
2. **Performance degrades with complexity**: More operations = lower accuracy
3. **Fine-tuned T5 can learn tracking**: Small models can acquire this capability
4. **Lexical overlap matters**: Generalization to new entities/operations is limited

#### Relevance to Our Research
- **Critical baseline**: Their box-moving task is directly relevant to character tracking
- **Methodology**: Synthetic data generation approach is applicable
- **Finding**: Code pretraining correlates with entity tracking - suggests computational abstraction matters

---

### Paper 2: Lost in the Middle: How Language Models Use Long Contexts
- **Authors**: Nelson F. Liu, Kevin Lin, John Hewitt, et al.
- **Year**: 2024
- **Source**: TACL 2024
- **File**: `papers/2307.03172_lost_in_the_middle.pdf`

#### Key Contribution
Demonstrates that LLMs have a U-shaped performance curve - they use information at the beginning and end of context better than information in the middle.

#### Methodology
- **Tasks**: Multi-document QA and key-value retrieval
- **Setup**: Place relevant information at different positions in context
- **Models**: GPT-3.5-Turbo, Claude-1.3, MPT-30B, LongChat-13B

#### Key Findings
1. **U-shaped performance curve**: Best at beginning/end, worst in middle
2. **Performance degrades with more documents**: Adding context can hurt
3. **Even long-context models struggle**: 32K context doesn't mean 32K is usable
4. **Position effects persist**: True for both retrieval and reasoning tasks

#### Implications for Character Tracking
- Characters introduced in the middle of a story may be tracked worse
- Order of character introduction matters
- More characters = more potential for "lost in the middle" effects

---

### Paper 3: RULER: What's the Real Context Size of Your Long-Context LMs?
- **Authors**: Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al.
- **Year**: 2024
- **Source**: COLM 2024
- **File**: `papers/2404.06654_ruler_benchmark.pdf`

#### Key Contribution
Proposes a comprehensive benchmark with four task categories beyond simple retrieval, including **Variable Tracking** - a proxy for entity/character tracking.

#### Task Categories
1. **Retrieval**: Extended needle-in-haystack variants
2. **Multi-hop Tracing**: Variable tracking across multiple hops
3. **Aggregation**: Counting, common words extraction
4. **Question Answering**: With distracting information

#### Variable Tracking Task
```
VAR X1 = 12345
VAR X2 = X1
VAR X3 = X2
Q: Find all variables assigned value 12345
A: X1, X2, X3
```

#### Key Findings
1. **All models degrade with length**: Even 128K context models fail at claimed lengths
2. **Variable tracking is hard**: Multi-hop chains especially challenging
3. **Only Gemini-1.5 and GPT-4 perform well** at longer contexts
4. **Claimed context size ≠ effective context size**

#### Relevance to Character Tracking
- Variable tracking directly analogous to character attribute tracking
- Multi-hop chains = tracking character state through multiple changes
- Benchmark methodology applicable to character experiments

---

### Paper 4: One Thousand and One Pairs (NoCha Benchmark)
- **Authors**: Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer
- **Year**: 2024
- **Source**: EMNLP 2024
- **File**: `papers/2406.16264_nocha_benchmark.pdf`

#### Key Contribution
Creates a benchmark of true/false claim pairs about recently published novels, testing global reasoning over book-length texts.

#### Methodology
- **Data**: 1,001 minimal pairs about 67 novels (49K-336K tokens)
- **Task**: Verify if a claim about the novel is true or false
- **Design**: Minimal pairs isolate specific facts (one true, one false)

#### Key Findings
1. **All models struggle**: Best model (GPT-4o) achieves only 55.8%
2. **No open-weight model beats random chance**
3. **Global reasoning hardest**: 41.6% accuracy vs 59.8% for sentence-level
4. **World-building is difficult**: Speculative fiction (complex worlds) is hardest

#### Implications
- **Character tracking in real narratives is extremely challenging**
- **Synthetic benchmarks overestimate capabilities**
- **World-building complexity** (multiple characters with attributes) is a key difficulty

---

### Paper 5: (How) Do Language Models Track State?
- **Authors**: Belinda Z. Li, Zifan Carl Guo, Jacob Andreas
- **Year**: 2025
- **Source**: ICML 2025
- **File**: `papers/2503.02854_how_do_lms_track_state.pdf`

#### Key Contribution
Investigates the **mechanisms** LLMs use to track state, using permutation composition as a model task.

#### Methodology
- **Task**: Compute final order of objects after sequence of swaps
- **Analysis**: Probing, attention analysis, intervention experiments
- **Models**: Various transformer architectures

#### Key Findings
1. **Two mechanisms discovered**:
   - **Associative Algorithm (AA)**: Resembles theoretical "associative scan"
   - **Parity-Associative Algorithm (PAA)**: Uses parity heuristic + scan
2. **Mechanism affects generalization**: AA generalizes better to longer sequences
3. **Training dynamics predict mechanism**: Can steer toward one or other
4. **Step-by-step simulation NOT found**: Models don't simulate state evolution layer-by-layer

#### Relevance
- **Theoretical grounding** for what mechanisms models might use for character tracking
- **Probing methodology** applicable to studying character representations
- **Suggests capacity limits** tied to specific algorithms, not just "memory"

---

### Paper 6: Too Long, Didn't Model (TLDM)
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv
- **File**: `papers/2505.14925_too_long_didnt_model.pdf`

#### Key Contribution
Decomposes long-context understanding to identify where in the context window models begin to fail.

#### Key Findings
1. **State-tracking requires integration across context**
2. **Mamba/S4 (state-space models) struggle with state tracking**
3. **NoCha shows near-random performance** for texts averaging 127K tokens
4. **Position-independent failures** for some models

---

### Paper 7: CharacterBench
- **Authors**: Jinfeng Zhou, Yongkang Huang, et al.
- **Year**: 2024
- **Source**: AAAI 2025
- **File**: `papers/2412.11912_character_bench.pdf`

#### Key Contribution
Largest benchmark for evaluating character customization in role-playing LLMs, with 22,859 samples covering 3,956 characters.

#### Dimensions Evaluated
1. **Memory**: Memory consistency
2. **Knowledge**: Fact accuracy, boundary consistency
3. **Persona**: Attribute consistency, behavior consistency
4. **Emotion**: Self-regulation, empathetic responsiveness
5. **Morality**: Moral robustness, stability
6. **Believability**: Human-likeness, engagement

#### Relevance
- **Comprehensive character attributes** to track
- **Evaluation framework** for multi-dimensional character assessment
- **Sparse vs. dense dimensions**: Some attributes manifest in every response, others rarely

---

## Common Methodologies Across Papers

### Synthetic Data Generation
- **Kim & Schuster**: Box-moving with objects
- **RULER**: Variable assignment chains
- **bAbI**: Location/object tracking scenarios

All use:
- Controlled number of entities
- Controlled number of state changes
- Programmatic generation for scale
- Clear ground truth for evaluation

### Evaluation Approaches
1. **Exact match accuracy**: Does predicted state match ground truth?
2. **Per-entity accuracy**: Break down by specific entities
3. **Complexity scaling**: Performance vs. number of operations/entities
4. **Position analysis**: Performance by location in context

---

## Standard Baselines

### For Entity Tracking
1. **bAbI Tasks** (especially 1-3): Classic toy benchmarks
2. **Kim & Schuster box task**: Entity tracking baseline
3. **RULER Variable Tracking**: Multi-hop tracking

### For Long-Context
1. **Needle-in-haystack**: Simple retrieval baseline
2. **Multi-document QA**: Realistic retrieval-reasoning
3. **NoCha**: Book-length comprehension

---

## Evaluation Metrics

### Primary Metrics
1. **Accuracy**: % correct state predictions
2. **Pair Accuracy** (NoCha): Both true and false claims correct
3. **F1 Score**: For multi-label predictions

### Analysis Metrics
1. **Accuracy vs. Number of Entities**: Scaling behavior
2. **Accuracy vs. Position**: U-curve analysis
3. **Accuracy vs. Number of Operations**: Complexity scaling
4. **Per-Entity Breakdown**: Which characters are tracked well?

---

## Datasets in the Literature

| Dataset | Task | Entities | Context Length | Source |
|---------|------|----------|----------------|--------|
| bAbI | QA | 2-5 | Short | Facebook |
| Kim & Schuster | State prediction | 7 boxes | Short | Authors |
| RULER VT | Variable tracking | 2-10 chains | 4K-128K | NVIDIA |
| NoCha | Claim verification | Novel characters | 49K-336K | Authors |
| NarrativeQA | QA | Multiple | Full novels | DeepMind |

---

## Gaps and Opportunities

### Gap 1: Systematic Character Count Studies
**No existing work systematically varies the number of characters** to find breaking points.
- Kim & Schuster use fixed 7 boxes
- RULER uses fixed chain patterns
- Our research can fill this gap

### Gap 2: Natural Language Character Tracking
Most studies use synthetic tasks with simple templates.
- Real stories have complex, naturalistic language
- Character references may be indirect (pronouns, descriptions)
- Our research can bridge synthetic and naturalistic

### Gap 3: Multi-Attribute Tracking
Existing work focuses on single attributes (location OR object).
- Characters have multiple simultaneous attributes
- Need to track location + mood + possessions + relationships
- Our synthetic dataset includes multi-attribute tracking

### Gap 4: Mechanistic Understanding
Li et al. (2025) analyze permutation tracking, but:
- Character tracking may use different mechanisms
- Need to study attention patterns over character mentions
- Probing for character-specific representations

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary**: Our synthetic character tracking dataset (varying 2-20 characters)
2. **Baseline**: bAbI Task 2 (two supporting facts) for sanity check
3. **Reference**: RULER Variable Tracking for comparison

### Recommended Baselines
1. **Random**: Always predict most common state
2. **First-mention**: Predict initial state (ignores changes)
3. **Last-mention**: Predict based on most recent mention only
4. **Full-context**: Proper model inference

### Recommended Metrics
1. **Overall accuracy** by number of characters
2. **Per-character accuracy** (do early/late characters differ?)
3. **Per-question-type accuracy** (location vs. mood vs. holding)
4. **Position analysis** (characters introduced early vs. late)

### Methodological Considerations
1. **Control for token count**: More characters = longer context
2. **Control for mentions**: Ensure each character mentioned similar times
3. **Use multiple seeds**: 3+ trials per configuration
4. **Test multiple models**: GPT-4, Claude, Llama, etc.

---

## Theoretical Framework

Based on the literature, we hypothesize:

1. **Capacity Limit Exists**: Models have a finite number of entity "slots"
2. **Mechanism-Dependent**: The tracking algorithm determines the limit
3. **Position Effects**: Middle characters may be tracked worse
4. **Code Pretraining Helps**: Computational abstraction improves tracking
5. **Complexity Matters**: More state changes = harder tracking

The experiment should test each of these hypotheses.

---

## Key References

1. Kim, N., & Schuster, S. (2023). Entity Tracking in Language Models. ACL.
2. Liu, N. F., et al. (2024). Lost in the Middle. TACL.
3. Hsieh, C.-P., et al. (2024). RULER. COLM.
4. Karpinska, M., et al. (2024). NoCha. EMNLP.
5. Li, B. Z., et al. (2025). How Do Language Models Track State? ICML.
6. Weston, J., et al. (2015). Towards AI-complete QA: bAbI tasks. arXiv.
