# Research Plan: Character Tracking Limits in Language Models

## Research Question

**How many characters can a language model keep track of in-context, and at what point does performance degrade?**

We aim to identify the breaking point where LLMs struggle to track multiple characters' states (location, mood, possessions) and use these failure patterns to gain insight into the mechanisms models use for entity tracking.

## Background and Motivation

Complex narratives often contain many characters (e.g., War and Peace, Tale of Genji). Understanding LLM limitations in character tracking has practical implications for:
1. AI-assisted writing and storytelling
2. Document understanding and summarization
3. Agent-based simulations involving multiple entities
4. Understanding fundamental attention/memory mechanisms in transformers

The literature (see literature_review.md) reveals:
- Entity tracking degrades with complexity (Kim & Schuster, 2023)
- Position effects exist - "lost in the middle" (Liu et al., 2024)
- Variable tracking in RULER shows multi-hop chains are hard (Hsieh et al., 2024)
- **Gap**: No systematic study varying character count from 2-20+

## Hypothesis Decomposition

### Primary Hypothesis
**H1**: There exists a capacity limit on the number of characters an LLM can simultaneously track, and performance will degrade as character count increases.

### Secondary Hypotheses
**H2**: Performance degradation is non-linear - there is a "cliff" or threshold where tracking ability sharply declines.

**H3**: Different question types (location vs. mood vs. holding) may have different capacity limits.

**H4**: Story complexity (number of actions/state changes) interacts with character count - more actions amplify degradation.

**H5**: More capable models (GPT-4 vs GPT-3.5) will have higher capacity limits.

## Proposed Methodology

### Approach
We use a synthetic character tracking dataset that systematically varies:
- Number of characters: 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
- Number of actions: 5, 10, 20
- Question types: location, mood, holding

This controlled design allows us to isolate the effect of character count while controlling for confounds.

### Experimental Steps

1. **Baseline Validation**: Verify models can perform basic entity tracking (2 characters, 5 actions)

2. **Character Count Scaling**: For each character count (2-20):
   - Present story with all character introductions and actions
   - Ask questions about each character's final state
   - Record accuracy per character and overall

3. **Action Count Interaction**: Test whether more actions (5 vs 10 vs 20) affects the character tracking limit

4. **Question Type Analysis**: Compare accuracy across location/mood/holding to identify if certain state types are harder to track

5. **Model Comparison**: Test multiple models:
   - GPT-4.1 (via OpenAI API)
   - GPT-3.5-turbo (via OpenAI API)
   - Claude Sonnet 4 (via OpenRouter)

### Baselines

1. **Random Baseline**: Predict random valid answers from the vocabulary
2. **First-State Baseline**: Always predict the initial state (ignores all changes)
3. **Last-Mention Heuristic**: Predict based on the most recent sentence about each character

### Evaluation Metrics

**Primary Metrics**:
- **Overall Accuracy**: % of questions answered correctly
- **Per-Character-Count Accuracy**: Accuracy grouped by number of characters in story

**Secondary Metrics**:
- **Per-Question-Type Accuracy**: Location vs mood vs holding
- **Per-Action-Count Accuracy**: How 5/10/20 actions affects performance
- **Character Position Effect**: Do characters introduced early vs late differ?

### Statistical Analysis Plan

- **Significance testing**: McNemar's test for paired comparisons between conditions
- **Trend analysis**: Linear regression of accuracy vs. character count to find slope
- **Effect sizes**: Cohen's d for differences between character count conditions
- **Multiple comparisons**: Bonferroni correction when comparing multiple models

## Expected Outcomes

### If H1 is supported:
- Accuracy decreases as character count increases
- Clear performance gap between low (2-4) and high (15-20) character stories

### If H2 is supported:
- Non-linear degradation curve with identifiable threshold (e.g., sharp drop at 8+ characters)

### Indicators of rejection:
- Flat accuracy regardless of character count (models have no practical limit)
- Random performance even at low character counts (task too hard in general)

## Timeline and Milestones

1. **Environment Setup & Data Loading** (5 min) - Complete
2. **Implement Experiment Code** (30 min)
3. **Run Experiments** (60 min - depends on API calls)
4. **Analyze Results** (30 min)
5. **Generate Visualizations** (15 min)
6. **Write Report** (30 min)

## Potential Challenges

1. **API Rate Limits**: May need to add delays between calls
   - Mitigation: Use tqdm for progress, exponential backoff

2. **Cost**: Many API calls
   - Mitigation: Start with subset, estimate costs before full run

3. **Variance**: Single trials may be noisy
   - Mitigation: Dataset has 3 trials per configuration (90 examples total)

4. **Model non-determinism**: Temperature > 0 adds noise
   - Mitigation: Use temperature=0 for reproducibility

## Success Criteria

1. **Quantitative**: Successfully measure accuracy across all character count levels
2. **Pattern Finding**: Identify clear trend or threshold in character tracking ability
3. **Statistical Validity**: Results statistically significant (p < 0.05)
4. **Model Comparison**: At least 2 models compared
5. **Reproducibility**: All results documented with exact prompts and parameters

## Resource Requirements

- **APIs**: OpenAI (GPT-4.1, GPT-3.5-turbo), OpenRouter (Claude)
- **Compute**: CPU only (API-based)
- **Estimated API Cost**: ~90 examples × 2-3 models × ~$0.01-0.05/example = $5-20
- **Time**: ~2-3 hours total

## Experimental Protocol

### Prompt Template
```
Read the following story and answer the question based on the final state of affairs.

Story:
{story}

Question: {question}

Answer with only the answer word (e.g., "kitchen", "happy", "book", or "nothing").
```

### Configuration
- Temperature: 0
- Max tokens: 20
- Random seed: 42 (where supported)
