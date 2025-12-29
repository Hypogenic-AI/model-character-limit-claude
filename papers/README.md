# Downloaded Papers

This directory contains research papers relevant to the character tracking limits study.

## Paper List

### 1. Entity Tracking in Language Models
- **File**: `2305.02363_entity_tracking_in_language_models.pdf`
- **Authors**: Najoung Kim, Sebastian Schuster
- **Year**: 2023
- **Venue**: ACL 2023
- **arXiv**: [2305.02363](https://arxiv.org/abs/2305.02363)
- **Why relevant**: First systematic study of entity tracking in LLMs; provides baseline methodology and box-moving task design

### 2. Lost in the Middle: How Language Models Use Long Contexts
- **File**: `2307.03172_lost_in_the_middle.pdf`
- **Authors**: Nelson F. Liu, Kevin Lin, John Hewitt, et al.
- **Year**: 2024
- **Venue**: TACL 2024
- **arXiv**: [2307.03172](https://arxiv.org/abs/2307.03172)
- **Why relevant**: Demonstrates U-shaped performance curve; important for understanding position effects in character tracking

### 3. RULER: What's the Real Context Size of Your Long-Context LMs?
- **File**: `2404.06654_ruler_benchmark.pdf`
- **Authors**: Cheng-Ping Hsieh, Simeng Sun, et al.
- **Year**: 2024
- **Venue**: COLM 2024
- **arXiv**: [2404.06654](https://arxiv.org/abs/2404.06654)
- **Why relevant**: Variable tracking task directly analogous to character tracking; provides evaluation framework

### 4. One Thousand and One Pairs (NoCha)
- **File**: `2406.16264_nocha_benchmark.pdf`
- **Authors**: Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer
- **Year**: 2024
- **Venue**: EMNLP 2024
- **arXiv**: [2406.16264](https://arxiv.org/abs/2406.16264)
- **Why relevant**: Shows models struggle with character tracking in real novels; establishes difficulty of global reasoning

### 5. (How) Do Language Models Track State?
- **File**: `2503.02854_how_do_lms_track_state.pdf`
- **Authors**: Belinda Z. Li, Zifan Carl Guo, Jacob Andreas
- **Year**: 2025
- **Venue**: ICML 2025
- **arXiv**: [2503.02854](https://arxiv.org/abs/2503.02854)
- **Why relevant**: Mechanistic analysis of state tracking; reveals algorithms models use

### 6. Too Long, Didn't Model
- **File**: `2505.14925_too_long_didnt_model.pdf`
- **Authors**: Various
- **Year**: 2025
- **Venue**: arXiv
- **arXiv**: [2505.14925](https://arxiv.org/abs/2505.14925)
- **Why relevant**: Decomposes long-context failures; identifies where models begin to fail

### 7. CharacterBench
- **File**: `2412.11912_character_bench.pdf`
- **Authors**: Jinfeng Zhou, Yongkang Huang, et al.
- **Year**: 2024
- **Venue**: AAAI 2025
- **arXiv**: [2412.11912](https://arxiv.org/abs/2412.11912)
- **Why relevant**: Comprehensive character evaluation framework with multiple dimensions

## Usage

```python
import pdfplumber

# Read paper
with pdfplumber.open("papers/2305.02363_entity_tracking_in_language_models.pdf") as pdf:
    for page in pdf.pages[:5]:  # First 5 pages
        text = page.extract_text()
        print(text)
```

## Citation Information

See `literature_review.md` for full citation details and synthesis of findings.
