<div style="text-align: left;">
  <img src="https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/figure/oppaWord_logo.png" width="300" alt="oppaWord Logo">
</div>

# oppaWord: Super Fast Myanmar Word Segmenter


ကျွန်တော်တို့ စာရေးတဲ့အခါမှာ အင်္ဂလိပ်စာလိုမျိုး စာလုံးတွေကို ဖြတ်ပြီးရေးလေ့မရှိတာကြောင့်၊ NLP/AI အလုပ်တွေအတွက် မြန်မာစာကြောင်းတွေကို စာလုံးဖြတ်ရတဲ့အလုပ်က တကယ်ခက်ခဲပါတယ်။ အဲဒါကြောင့် ကျွန်တော်ဦးဆောင်နေတဲ့ Language Understanding Lab. မှာ Word Segmentation သုတေသနအလုပ်တွေကို အချိန်ရရင်ရသလို ဆက်တိုက်လုပ်ဖြစ်နေပါတယ်။ တချို့အလုပ်တွေအတွက် [sylbreak](https://github.com/ye-kyaw-thu/sylbreak) နဲ့ ဝဏ္ဏဖြတ်ပြီး လုပ်တယ်၊ တချို့အလုပ်တွေအတွက် [myWord](https://github.com/ye-kyaw-thu/myWord) နဲ့ စာလုံးဖြတ်တယ်။ ဒါ့အပြင် [ငါးပိ](https://github.com/ye-kyaw-thu/NgaPi)လို့ နာမည်ပေးထားတဲ့ semantic chunking လိုမျိုး စာလုံးတွေကို ဗက်တာပြောင်းပြီး အဓိပ္ပါယ်ပါ ဆွဲယူဖို့ ဖြစ်နိုင်တဲ့ စာလုံးဖြတ်နည်းတွေကိုလည်း လေ့လာစမ်းသပ်ဖြစ်ခဲ့ပါတယ်။ ဒီရက်ပိုင်းတွေအတွင်းမှာ Lab ရဲ့ internship ကျောင်းသားတွေကို word segmentation နဲ့ပတ်သက်တာလည်း စာသင်ဖို့ပြင်ရင်းနဲ့ ဟိုးအရင်က စမ်းခဲ့ဖူးတဲ့ DAG (directed acyclic graph) ဘက်ကို ပြန်လှည့်ဖြစ်ခဲ့ပါတယ်။ အိုက်ဒီရာအနေနဲ့ကတော့ ARPA n-gram language model ရယ် syllable frequency count တွေရယ်နဲ့ မြန်မာစာ စာလုံးတွေကို ဖြတ်တဲ့ နည်းလမ်းပါ။ သူက OOV ကိုလည်း backoff နဲ့ ရှောင်လို့ ရတာမို့ ကျွန်တော်က စိတ်ဝင်စားတယ်။ လက်တွေ့မှာက မြန်မာစာအတွက် တကယ်ကောင်းတဲ့ language model ဆောက်ဖို့ဆိုတာက မလွယ်ကူပါဘူး။ အမျိုးမျိုး ကြိုးစားကြည့်လည်း ရလဒ်က ထင်သလောက် တက်မလာပါဘူး။ အဲဒါနဲ့ နောက်ဆုံးတော့ အဘိဓာန် နဲ့ bidirectional maximum matching ကိုပါ တွဲလိုက်ပြီး score လုပ်တာ fallback လုပ်တာတွေနဲ့ တွဲဖြစ်သွားပါတယ်။ ရလဒ်က --bimm-boost နဲ့ tuning လုပ်ရင်း F1-score 90+ ထိ တက်အောင် လုပ်လို့ရတာကို ရှာဖွေတွေ့ရှိခဲ့တယ်။ အဲဒါနဲ့ပဲ ကျောင်းသား၊ ကျောင်းသူတွေကိုလည်း Hybrid DAG + Bi-MM + LM architecture ကိုအခြေခံတဲ့ word segmentation ကို မိတ်ဆက်ပေးရင်း ဒီ oppaWord ကို အများသုံးလို့ရဖို့အထိ coding ဆက်လုပ်သွားဖြစ်ခဲ့တယ်။  

oppaWord ဆိုတဲ့ နာမည်လား။ အစက ARPA LM ကို သုံးထားတာမို့လို့ arpaWord ဆိုပြီး နာမည်ပေးဖို့ စဉ်းစားခဲ့တာ။ တကယ်တမ်းက hybrid_dag_bimm_lm.py ဆိုပြီး ပေးရင်တော့ သုံးထားတဲ့ approach အားလုံးလိုလိုကို ခြုံငုံမိပါတယ်။ ခက်တာက အများအတွက် မှတ်မိဖို့ ခက်ပါလိမ့်မယ်။ လက်ရှိ ကိုရီးယား စကားလုံး [oppa](https://en.wiktionary.org/wiki/%EC%98%A4%EB%B9%A0) (오빠) ဆိုရင်တော့ မြန်မာလူငယ်တိုင်းလိုလို ရင်းနှီးပြီးသား ဖြစ်ပါလိမ့်မယ်။ ပြီးတော့ ၁၀တန်းအပြီးမှာ နှစ်တော်တော်ကြာကြာ တိုက်ကွမ်ဒိုအားကစားကို လုပ်ဖြစ်ခဲ့တဲ့ ငယ်ဘဝ အမှတ်တရတွေရယ်၊ ကျွန်တော်ရဲ့ တိုက်ကွမ်ဒိုဆရာတွေအများကြီးထဲကတစ်ဦးဖြစ်တဲ့ ကမ္ဘာ့ချန်ပီယံ ဆရာဂျွန် ကိုလည်း သတိရတာနဲ့ oppaWord လို့ပဲ နာမည်ပေးဖြစ်ခဲ့ပါတယ်။  

လက်ရှိအချိန်ထိ အမျိုးမျိုး experiment တွေ လုပ်ကြည့်ခဲ့ပြီး စာကြောင်းရေ လေးသောင်းကျော်ကို (e.g. myPOS corpus တစ်ခုလုံး) စာလုံးဖြတ်ကြည်တာ ၃ စက္ကန့်တောင် မကြာပါဘူး။ Language model ဖြည့်ပြီး run ရင်တော့ စက္ကန့်အနည်းငယ် ပိုကြာပါလိမ့်မယ်။ အဲဒါကြောင်း လက်ရှိအချိန်ထိ **မြန်မာစာအတွက် အမြန်ဆုံး word segmenter** ပါပဲ။ **오ppa빠ord** ကို အားပေးကြပါဦးလို့။  

ရဲကျော်သူ  
4 Aug 2025

## Overview

Myanmar language lacks strict word boundary rules, making word segmentation essential for NLP tasks. While existing tools like [sylbreak](https://github.com/ye-kyaw-thu/sylbreak) (syllable segmenter) and [myWord](https://github.com/ye-kyaw-thu/myWord) (multi-level segmenter) exist, **oppaWord** fills the critical need for a **fast**, **training-free** word segmenter with domain adaptation capabilities through:

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
$ python ./oppa_word.py --help
usage: oppa_word.py [-h] --input INPUT [--output OUTPUT] --dict DICT [--sylfreq SYLFREQ] [--arpa ARPA]
                    [--postrule-file POSTRULE_FILE] [--max-order MAX_ORDER] [--dict-weight DICT_WEIGHT]
                    [--use-bimm-fallback] [--bimm-boost BIMM_BOOST] [--visualize-dag] [--dag-output-dir DAG_OUTPUT_DIR]
                    [--space-remove-mode {all,my,my_not_num}] [--max-word-len MAX_WORD_LEN]

oppa_word, Hybrid DAG + BiMM + LM Myanmar Word Segmenter with optional Aho-Corasick support

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input file with one sentence per line (UTF-8)
  --output OUTPUT, -o OUTPUT
                        Optional output file path (default: stdout)
  --dict DICT, -d DICT  Word dictionary file (one word per line)
  --sylfreq SYLFREQ, -s SYLFREQ
                        Syllable frequency file (syllable<TAB>frequency, for scoring)
  --arpa ARPA, -a ARPA  ARPA-format syllable-level language model (optional)
  --postrule-file POSTRULE_FILE
                        Optional post-processing rules (e.g., merging, corrections)
  --max-order MAX_ORDER
                        Max LM n-gram order (default: 5)
  --dict-weight DICT_WEIGHT
                        Dictionary path weight in scoring (default: 10.0)
  --use-bimm-fallback   Enable Bi-directional Maximum Matching as fallback
  --bimm-boost BIMM_BOOST
                        Boost score added to Bi-MM fallback path (default: 0.0)
  --visualize-dag       Generate DAG visualization (PDF per sentence)
  --dag-output-dir DAG_OUTPUT_DIR
                        Directory to save DAG PDFs if --visualize-dag is used (default: 'dag_viz')
  --space-remove-mode {all,my,my_not_num}
                        Preprocessing mode to remove spaces: 'all', 'my' (Myanmar only), or 'my_not_num (Myanmar but not
                        including Myanmar numbers'
  --max-word-len MAX_WORD_LEN
                        Maximum word length in syllables (3-12, default:6)
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
├── data/ # Resource files
│   ├── myg2p_mypos.dict # Main dictionary
│   ├── myg2p_mypos_name.dict # Extended dictionary with Myanmar names
│   ├── myMono.freq # Syllable frequency counts
│   ├── myMono_clean_syl.trie.bin # 5-gram syllable LM (optimized binary)
│   └── rules.txt # Post-processing correction rules
├── doc/ # Documentation
├── tools/ # Evaluation and preprocessing scripts
└── oppa_word.py # Main segmenter code
```

Note: [exp_1](https://github.com/ye-kyaw-thu/oppaWord/tree/main/exp_1), [exp_2](https://github.com/ye-kyaw-thu/oppaWord/tree/main/exp_2), and [exp_20250727_1553](https://github.com/ye-kyaw-thu/oppaWord/tree/main/exp_20250727_1553) are output folders from some earlier experiments.  

## Data Preparation  

### Input Format  

- One sentence per line
- Space removal optional (handled by `--space-remove-mode`)

## Post-Editing Rules (rules.txt)

oppaWord supports post-segmentation corrections through a rules file. This helps fix systematic errors and improve readability.

### Rule Format

SOURCE|||TARGET  

### Types of Rules

1. Exact Word Replacements:
   ```text
   ပါတယ်|||ပါ တယ်
   မရှိ|||မ ရှိ
   ```
   - Merges or splits exact word matches
   - Example: `"ပါတယ်"` → `"ပါ တယ်"`

2. Regular Expressions:

   ```text
   (\S)([၊။])|||\1 \2
   ```
   - Uses regex patterns to handle:
     - Punctuation attachment (`"ပါ။"` → `"ပါ ။"`)
   - Regex syntax follows Python's re module

3. Best Practices

   - Order Matters: Rules are applied top-to-bottom
   - Balance Specificity:  
     - Prefer exact matches (`ပါဘူး|||ပါ ဘူး`) over broad regex when possible  
  
## Performance Tips

1. For speed: Use `--dict-only --bimm-boost 150`
2. For accuracy: Add `--arpa data/myMono_clean_syl.trie.bin`
3. Domain adaptation:
   - Customize `data/rules.txt` for post-editing
   - Add domain terms to dictionary
  
## License

### Source Code & Tools
**MIT License** - Full terms available at:  
[https://github.com/ye-kyaw-thu/oppaWord/blob/main/LICENSE](https://github.com/ye-kyaw-thu/oppaWord/blob/main/LICENSE)

### Dictionary Data

The dictionary combines words from multiple sources:  

- myG2P dictionary (originally from Myanmar Language Commission)
- myPOS corpus
- Personal names from myRoman corpus and LU Lab's Myanmar names collection (for R&D)
- myMono monolingual corpus (not publicly released)

**Usage Restrictions**:
- ✅ **Allowed**: Myanmar language NLP/AI research and development
- ❌ **Not Allowed**: Commercial use without explicit permission

## Detailed Technical Information

For those interested in oppaWord's segmentation methodology and development process:  

1. **Technical Introduction**  
   - Presentation slides:  
     [oppaWord_Intro.pdf](https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/talk/oppaWord_Intro.pdf)  
     [oppaWord_Intro.pptx](https://github.com/ye-kyaw-thu/oppaWord/blob/main/doc/talk/oppaWord_Intro.pptx)  
   - Covers: Core algorithm, visualization examples, and performance benchmarks

2. **Experimental Notebooks**  
   [![Open Notebooks](https://img.shields.io/badge/Explore-Notebooks-blue)](https://github.com/ye-kyaw-thu/oppaWord/tree/main/notebook)  
   Contains Jupyter notebooks demonstrating:
   - Different segmentation strategies
   - Error analysis workflows
   - Parameter tuning examples

3. **Research Publications** *(Coming Soon)*  
   - Planned journal paper on the hybrid segmentation approach

## Citation  

If you use oppaWord in your work, please cite it as follows:  
*(oppaWord ကို အသုံးပြုပါက အောက်ပါအတိုင်း ကိုးကားဖော်ပြပေးပါ။)*  

```bibtex
@misc{oppaWord_2025,
  author       = {Ye Kyaw Thu},
  title        = {{oppaWord: Hybrid DAG+Bi-MM+LM Myanmar Word Segmenter}},
  version      = {1.0},
  month        = {August},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/ye-kyaw-thu/oppaWord},
  note         = {Accessed: YYYY-MM-DD},
  institution  = {Language Understanding Lab (LU Lab), Myanmar}
}
