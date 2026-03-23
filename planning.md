# Research Plan: How Many Characters Can a Model Keep Track Of?

## Motivation & Novelty Assessment

### Why This Research Matters
Language models are increasingly used for narrative understanding, summarization, and creative writing — all tasks requiring tracking multiple characters and their attributes. Understanding the limits of character tracking reveals fundamental constraints on LLM reasoning and has practical implications for complex narrative applications.

### Gap in Existing Work
The literature review reveals a critical gap: **no study systematically varies the number of characters** while holding other factors constant. Kim & Schuster (2023) fix at 7 boxes; RULER uses fixed chain patterns; TLDM tests real novels where character count and text length co-vary. We lack a clean measurement of how accuracy degrades as a function of character count.

### Our Novel Contribution
1. **First systematic character-count scaling study**: Vary characters from 2 to 32 in controlled synthetic narratives
2. **Error taxonomy**: Distinguish confusion errors (swapping attributes) from forgetting errors (reverting to initial state)
3. **Multi-model comparison on 2025 frontier models**: GPT-4.1 plus models via OpenRouter
4. **Attribute complexity interaction**: Test whether limits are per-character or per-attribute-slot

### Experiment Justification
- **Exp 1 (Character Count Scaling)**: Core measurement — the degradation curve. No prior work provides this.
- **Exp 2 (Error Type Analysis)**: Determines *how* models fail, revealing tracking mechanism.

## Research Question
Given a synthetic narrative with N characters, each with tracked attributes that change, how does accuracy at reporting character states degrade as N increases? What error patterns emerge?

## Hypothesis Decomposition
- H1: Accuracy degrades monotonically with character count, with a sharp transition.
- H2: Confusion errors (swapping attributes between characters) dominate over forgetting errors at high N.
- H3: Characters mentioned less frequently or introduced later are tracked worse.

## Proposed Methodology

### Story Generation
- **Character counts**: 2, 4, 8, 16, 32
- **Attributes per character**: 2 (location, possession)
- **State changes**: 2 per character (so total events scale with N)
- **Stories per condition**: 5 (for confidence intervals)
- **Names**: Distinctive, unambiguous names
- **Ground truth**: Programmatically verified

### Models to Test
- GPT-4.1 (OpenAI API) — frontier model
- GPT-4.1-mini (OpenAI API) — smaller frontier
- At least one more via OpenRouter if available

### Baselines
1. Random: guess uniformly from mentioned values
2. Initial-state: always predict starting attribute
3. Most-recent-mention

### Evaluation Metrics
- Overall accuracy per character count
- Per-character accuracy
- Confusion rate: wrong answer belongs to another character
- Forgetting rate: wrong answer is the initial state
- Position effect: accuracy by introduction order

### Statistical Analysis
- Bootstrap 95% CIs for accuracy
- Spearman correlation: character count vs accuracy

## Expected Outcomes
- Near-perfect at 2-4 characters, significant degradation by 16-32
- Confusion errors increase with character count
- Clear degradation curve suitable for identifying transition region

## Timeline
1. Setup + story generator: 20 min
2. API experiments: 60 min
3. Analysis + visualization: 30 min
4. Documentation: 20 min

## Success Criteria
- Clear degradation curve with CIs across ≥5 character counts
- Error taxonomy with confusion vs forgetting rates
- Comparison across ≥2 models
- Statistical significance of main findings
