<div style="text-align: left;">
  <img src="https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/figure/oppaWord_logo.png" width="300" alt="oppaWord Logo">
</div>

# oppaWord: Fast Domain-Adaptable Myanmar Word Segmenter

## Overview

Myanmar language lacks strict word boundary rules, making word segmentation essential for NLP tasks. While existing tools like `[sylbreak](https://github.com/ye-kyaw-thu/sylbreak)` (syllable segmenter) and `[myWord](https://github.com/ye-kyaw-thu/myWord)` (multi-level segmenter) exist, **oppaWord** fills the critical need for a **fast**, **training-free** word segmenter with domain adaptation capabilities through:

- Hybrid DAG + Bi-MM + LM architecture
- Post-editing rule support
- Visual debugging tools
- Syllable-level processing

**Key Advantages**:
- Faster than Neural Network based segmenters
- No training required - just provide a dictionary
- Tunable segmentation strategies via simple parameters

## Algorithm Explained

<div style="text-align: center;">
  <img src="https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/figure/overview-of-oppaWord.png" width="1000" alt="Overview of oppaWord Segmenter">
</div>

oppaWord combines three core techniques:

1. **DAG Construction**:
   - Builds all possible segmentations (3-12 syllable lengths)
   - Scores paths using dictionary, frequency, and language model features

2. **Bidirectional Maximum Matching (Bi-MM)**:
   - Fallback mechanism when DAG paths are uncertain
   - Configurable score boosting (`--bimm-boost`)

3. **Multi-Feature Scoring**:
   ```python
   Total_Score = Dict_Weight + Syllable_Freq + LM_Score + (Bi-MM_Boost)
   ```

## Installation

```
git clone https://github.com/ye-kyaw-thu/oppaWord.git  
cd oppaWord  
7z x data/5gramLM.7z.001  # Extract language model
7z x data/5gramLM.7z.002
7z x data/5gramLM.7z.003
```

## Usage
### Basic Command

```
python oppa_word.py \
  --input input.txt \
  --dict data/myg2p_mypos.dict \
  --output segmented.txt
```








