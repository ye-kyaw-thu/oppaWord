# oppaWord
## Algorithm Explanation

## **1. Dictionary-Only Segmentation**
**Command:** `python oppaWord.py -i input.txt -w words.txt --no-sylbreak`

### **How it Works**
- Uses **exact dictionary matching** to identify words.
- Implements a **dynamic programming (DP) approach** to find the best segmentation path.
- Scores segmentations based on whether words exist in the dictionary.

### **Example**
**Input (already syllable-segmented):**  
`ကလေး မီး ဖွား ဖို့`  
*(Equivalent to "for the child to be born")*

**Dictionary (`words.txt`):**  
```
ကလေး
မီးဖွား
ဖို့
မီး
ဖွား
```

### **Step-by-Step Calculation**
1. **DP Table Initialization**
   - `dp[0] = {'score': 0, 'path': []}` (empty prefix)
   - `dp[i]` tracks the best segmentation up to the *i-th* syllable.

2. **Possible Segmentations & Scores**
   - `ကလေး` (exists in dict → score = 1.0)  
     → `dp[1] = {'score': 0.7, 'path': ['ကလေး']}`  
     *(λ=0.7 for dictionary weight)*
   - `ကလေး + မီး` (OOV → score = 0.0)  
     → Not selected (lower score than alternatives).
   - `ကလေး + မီးဖွား` (both in dict → score = 0.7 + 0.7 = 1.4)  
     → `dp[3] = {'score': 1.4, 'path': ['ကလေး', 'မီးဖွား']}`
   - `ကလေး + မီး + ဖွား` (all in dict → score = 0.7 + 0.7 + 0.7 = 2.1)  
     → But `မီးဖွား` is preferred because it merges syllables into a longer known word.

3. **Final Segmentation**  
   `['ကလေး', 'မီးဖွား', 'ဖို့']`  
   *(Score = 0.7 + 0.7 + 0.7 = 2.1)*

---

## **2. Dictionary + Syllable Frequency**
**Command:** `python oppaWord.py -i input.txt -w words.txt -s syl_freq.txt --no-sylbreak`

### **How it Works**
- Still uses dictionary matching, but **adds syllable frequency scores** for Out-of-Vocabulary (OOV) words.
- Combined score:  
  **`score = λ * dict_score + (1-λ) * log(syllable_frequency)`**

### **Example**
**Input:** `ကလေး မွေးဖွား ဖို့`  
*(Note: `မွေးဖွား` is not in the dictionary, but its syllables are frequent.)*

**Syllable Frequencies (`syl_freq.txt`):**  
```
မွေး    1000
ဖွား    800
```

### **Step-by-Step Calculation**
1. **DP Table Update**
   - `ကလေး` (dict word → score = 0.7)
   - `မွေးဖွား` (OOV → syllable-based scoring):
     - `မွေး` (log(1000) ≈ 6.907)  
     - `ဖွား` (log(800) ≈ 6.684)  
     - **LM score** = (6.907 + 6.684) / 2 ≈ 6.795  
     - **Combined score** = (0.0 * 0.7) + (6.795 * 0.3) ≈ **2.038**
   - `ဖို့` (dict word → +0.7)

2. **Final Segmentation**  
   `['ကလေး', 'မွေးဖွား', 'ဖို့']`  
   *(Score = 0.7 + 2.038 + 0.7 = 3.438)*  
   *(Even though `မွေးဖွား` is OOV, its syllables are frequent enough to be accepted.)*

---

## **3. Dictionary + Syllable Frequency + ARPA LM**
**Command:** `python oppaWord.py -i input.txt -w words.txt -s syl_freq.txt --arpa 5gram.arpa --max-order 5 --no-sylbreak`

### **How it Works**
- Uses **n-gram probabilities** to evaluate word sequences.
- **Backoff smoothing** handles unseen n-grams.
- Scores:  
  **`score = λ * dict_score + (1-λ) * LM_probability`**

### **Example**
**Input:** `ကလေး မွေးဖွား ဖို့`  
**ARPA LM (`5gram.arpa`):**  
```
-1.0     ကလေး မွေးဖွား    -0.5
-2.0     မွေးဖွား ဖို့      -0.3
```

### **Step-by-Step Calculation**
1. **DP Table Update**
   - `ကလေး` (dict word → 0.7)
   - `မွေးဖွား` (OOV → check LM):
     - Bigram `ကလေး မွေးဖွား` → logprob = -1.0, backoff = -0.5  
     - **LM score** = -1.0  
     - **Combined score** = (0.0 * 0.7) + (-1.0 * 0.3) = **-0.3**
   - `ဖို့` (dict word → 0.7 + trigram score):
     - Trigram `ကလေး မွေးဖွား ဖို့` → Not found → Back off to bigram `မွေးဖွား ဖို့` → logprob = -2.0  
     - **LM score** = -2.0  
     - **Combined score** = (0.7 * 0.7) + (-2.0 * 0.3) ≈ **0.49 - 0.6 = -0.11**

2. **Final Segmentation**  
   `['ကလေး', 'မွေးဖွား', 'ဖို့']`  
   *(Score = 0.7 - 0.3 - 0.11 ≈ 0.29)*  
   *(LM prefers this sequence because `မွေးဖွား ဖို့` is a known bigram.)*

---

## **Summary Table**
| Method                     | Scoring Formula                          | Handles OOV? | Context-Aware? |
|----------------------------|------------------------------------------|--------------|----------------|
| **Dictionary Only**        | `λ * dict_score`                         | ❌ No        | ❌ No          |
| **+ Syllable Frequency**   | `λ * dict_score + (1-λ) * log(syl_freq)` | ✅ Yes       | ❌ No          |
| **+ ARPA LM**              | `λ * dict_score + (1-λ) * LM_prob`       | ✅ Yes       | ✅ Yes         |

---

### **Key Takeaways**
1. **Dictionary-only** is fast but fails on OOV words.
2. **Syllable frequencies** help guess OOV words.
3. **ARPA LM** improves accuracy by considering word context.
4. **λ (lambda)** controls the tradeoff between dictionary and statistical scores.

