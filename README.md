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
```

## Usage
### Basic Command

```
python oppa_word.py \
  --input input.txt \
  --dict data/myg2p_mypos.dict \
  --output segmented.txt
```

### Recommended Configuration

For best accuracy with current 5-gram LM:  

```
python oppa_word.py \
  --input text.txt \
  --dict data/myg2p_mypos.dict \
  --arpa data/myMono_clean_syl.trie.bin \
  --use-bimm-fallback \
  --bimm-boost 150 \
  --space-remove-mode "my_not_num"
```

### Full Options

```
$ python oppa_word.py --help
usage: oppa_word.py [-h] --input INPUT [--output OUTPUT] --dict DICT [--sylfreq SYLFREQ] 
                    [--arpa ARPA] [--postrule-file POSTRULE_FILE] [--max-order MAX_ORDER]
                    [--dict-weight DICT_WEIGHT] [--use-bimm-fallback] [--bimm-boost BIMM_BOOST]
                    [--visualize-dag] [--dag-output-dir DAG_OUTPUT_DIR]
                    [--space-remove-mode {all,my,my_not_num}] [--max-word-len MAX_WORD_LEN]

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input file (one sentence per line, UTF-8)
  --dict DICT, -d DICT  Dictionary file (one word per line)
  --arpa ARPA, -a ARPA  ARPA-format syllable LM
  --use-bimm-fallback   Enable Bi-MM fallback
  --bimm-boost BIMM_BOOST
                        Bi-MM path score boost (default: 0.0)
  --visualize-dag       Generate DAG visualizations
```

## Visualization

Debug segmentation decisions using DAG visualizations:  

```
python oppa_word.py \
  --input ./data/10lines.ref \
  --dict ./data/myg2p_mypos.dict \
  --space-remove-mode "my_not_num" \
  --use-bimm-fallback \
  --bimm-boost 150 \
  --visualize-dag \
  --dag-output-dir debug_viz2
```

Visualization Demo:  

```
ye@lst-hpc3090:~/exp/myTokenizer/oppaWord$ python oppa_word.py \
>   --input ./data/10lines.ref \
>   --dict ./data/myg2p_mypos.dict \
>   --space-remove-mode "my_not_num" \
>   --use-bimm-fallback \
>   --bimm-boost 150 \
>   --visualize-dag \
>   --dag-output-dir debug_viz2
၁၉၆၂ ခုနှစ် ခန့်မှန်း သန်းခေါင်စာရင်း အရ လူဦး ရေ ၁၁၅၉၃၁ ယောက် ရှိ သည်
လူ တိုင်း တွင် သင့်မြတ် လျော်ကန် စွာ ကန့်သတ် ထား သည့် အလုပ်လုပ်ချိန် အပြင် လစာ နှင့်တကွ အခါ ကာလ အားလျော်စွာ သတ်မှတ် ထား သည့် အလုပ် အားလပ်ရက် များ ပါဝင် သည့် အနားယူခွင့် နှင့် အားလပ်ခွင့် ခံစားပိုင်ခွင့် ရှိ သည်
ဤ နည်း ကို စစ်ယူ သော နည်း ဟု ခေါ် သည်
စာပြန်ပွဲ ဆို တာ က အာဂုံဆောင် အလွတ်ကျက် ထား တဲ့ ပိဋကတ်သုံးပုံ စာပေ တွေ ကို စာစစ် သံဃာတော်ကြီး တွေ ရဲ့ ရှေ့မှာ အလွတ် ပြန် ပြီး ရွတ်ပြ ရ တာ ပေါ့
ဒီ မှာ ကျွန်တော့် သက်သေခံကတ် ပါ
၂၀ ရာစု မြန်မာ့ သမိုင်း သန်း ဝင်း လှိုင် ၂၀၀၉ ခု မေ လ ကံကော်ဝတ်ရည် စာပေ
ကျွန်တော် မျက်မှန် တစ် လက်လုပ် ချင်ပါတယ်
ကျွန်တော် တို့ က ဒီ အမှု ရဲ့ ကြံရာပါ ကို ဖမ်းမိ ဖို့ ကြိုးစား ခဲ့ တယ်
ကလေး မီးဖွား ဖို့ ခန့်မှန်း ရက် က ဘယ်တော့ ပါ လဲ
အရိုးရှင်းဆုံး ကာဗိုဟိုက်ဒရိတ် မှာ ဂလူးကို့စ် ဂလက်တို့စ် ဖရပ်တို့စ် စသည့် မိုနိုဆက်ကရိုက် များ ဖြစ် သည်
ye@lst-hpc3090:~/exp/myTokenizer/oppaWord$ ls debug_viz2/
dag_line_0000.dot  dag_line_0002.dot  dag_line_0004.dot  dag_line_0006.dot  dag_line_0008.dot
dag_line_0000.pdf  dag_line_0002.pdf  dag_line_0004.pdf  dag_line_0006.pdf  dag_line_0008.pdf
dag_line_0001.dot  dag_line_0003.dot  dag_line_0005.dot  dag_line_0007.dot  dag_line_0009.dot
dag_line_0001.pdf  dag_line_0003.pdf  dag_line_0005.pdf  dag_line_0007.pdf  dag_line_0009.pdf
ye@lst-hpc3090:~/exp/myTokenizer/oppaWord$
```

Example output:  

<div style="text-align: center;">
  <img src="https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/figure/dag_line_0004.png" width="1000" alt="dag_line_0004.png">
</div>

## File Structure

```
oppaWord/
├── data/               # Resource files
│   ├── myg2p_mypos.dict       # Main dictionary
│   ├── myMono.freq            # Syllable frequencies  
│   └── myMono_clean_syl.arpa  # 5-gram LM
├── doc/                # Documentation
├── tools/              # Evaluation scripts
└── oppa_word.py        # Main segmenter code
```

Note: [exp_1](https://github.com/ye-kyaw-thu/oppaWord/tree/main/exp_1), [exp_2](https://github.com/ye-kyaw-thu/oppaWord/tree/main/exp_2), and exp_20250727_1553 are output folders from some earlier experiments.
