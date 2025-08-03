#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oppa_word, Hybrid DAG + Bi-MM + LM Myanmar Word Segmenter with Visualization

Core Features:
- DAG construction from dictionary and substrings (configurable length, default: ≤6 syllables)
- Bi-directional Maximum Matching (Bi-MM) fallback path
- Multi-feature scoring: Dictionary weight, syllable frequency, and ARPA/binary LM
- Advanced post-editing with regex and string replacement rules
- Configurable boosting for Bi-MM paths
- Integrated smart space removal with Myanmar-specific modes
- Punctuation-aware segmentation (၊ ။)

Optional Features:
- DAG visualization (.dot + .pdf with Graphviz)
- Binary LM support (KenLM format)
- Adjustable max n-gram order for LM scoring

Author: Ye Kyaw Thu, LU Lab., Myanmar
Date: 22 July 2025
Last Update: 26 July 2025
"""

import argparse
import re
import os
import sys
import math
import subprocess
from collections import defaultdict

try:
    import kenlm
    HAS_KENLM = True
except ImportError:
    HAS_KENLM = False

# === Unicode Character Classes for Space Removal ===
MYANMAR_LETTER = r'[\u1000-\u109F\uAA60-\uAA7F]'
MYANMAR_DIGIT = r'[\u1040-\u1049]'

# === Regex Patterns for Space Removal ===
RE_MM_LETTER_SPACE = re.compile(rf'({MYANMAR_LETTER})\s+({MYANMAR_LETTER})')
RE_MM_DIGIT_DIGIT = re.compile(rf'({MYANMAR_DIGIT})\s+({MYANMAR_DIGIT})')
RE_MM_DIGIT_LETTER = re.compile(rf'({MYANMAR_DIGIT})\s+({MYANMAR_LETTER})')
RE_MM_LETTER_DIGIT = re.compile(rf'({MYANMAR_LETTER})\s+({MYANMAR_DIGIT})')

PROTECT_SPACES = [
    (RE_MM_DIGIT_DIGIT, r'\1☃\2'),           # protect digit-digit
    (RE_MM_DIGIT_LETTER, r'\1☃\2'),          # protect digit-letter
    (RE_MM_LETTER_DIGIT, r'\1☃\2'),          # protect letter-digit
]


class HybridDAGSegmenter:
    def __init__(self, dict_path, syl_freq_path=None, arpa_lm_path=None,
                 max_order=5, dict_weight=10.0, postrule_file=None,
                 use_bimm_fallback=False, bimm_boost=0.0,
                 visualize_dag=False, dag_output_dir='dag_viz',
                 space_remove_mode=None, max_word_len=6):
        self.word_dict = self._load_dict(dict_path)
        self.syl_freq = self._load_freq(syl_freq_path) if syl_freq_path else {}
        self.lm = self._load_lm(arpa_lm_path) if arpa_lm_path else {}  # Changed method name
        self.max_order = max_order
        self.max_word_len = max(3, min(12, max_word_len))  # Enforce 3-12 range
        self.unk_logprob = -20.0
        self.break_pattern = self._create_break_pattern()
        self.dict_weight = dict_weight
        self.post_rules = self._load_post_rules(postrule_file) if postrule_file else []
        self.use_bimm_fallback = use_bimm_fallback
        self.bimm_boost = bimm_boost
        self.visualize_dag = visualize_dag
        self.dag_output_dir = dag_output_dir
        self.space_remove_mode = space_remove_mode
        os.makedirs(self.dag_output_dir, exist_ok=True)

    def _load_dict(self, path):
        return set(line.strip() for line in open(path, encoding='utf-8') if line.strip())

    def _load_freq(self, path):
        freq = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        syl, count = parts if not parts[0].isdigit() else (parts[1], parts[0])
                        freq[syl] = int(count)
                    except:
                        continue
        return freq

    def _load_lm(self, path):
        """Load LM from either ARPA or binary format"""
        if path.endswith('.bin') or path.endswith('.klm'):
            if not HAS_KENLM:
                raise ImportError("kenlm package required for binary LM support. Install with: pip install kenlm")
            return kenlm.Model(path)
        else:
            return self._load_arpa_lm(path)

    def _load_arpa_lm(self, path):
        """Load ARPA format LM (updated to handle binary detection)"""
        # Check if this is actually a binary file
        with open(path, 'rb') as f:
            if f.read(2) == b'\x00\x00':  # Simple binary detection
                raise ValueError("File appears to be binary. Use .bin extension for binary LMs")
        
        lm = {}
        current_order = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("\\") and "-grams:" in line:
                    try:
                        order_str = line.strip('\\').split('-')[0]
                        current_order = int(order_str)
                    except ValueError:
                        current_order = 0
                    continue
                if not line or line.startswith("\\") or line == "\\end\\":
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        logprob = float(parts[0])
                        ngram = parts[1]
                        lm[ngram] = logprob
                    except ValueError:
                        continue
        return lm

    def _load_post_rules(self, rule_file):
        rules = []
        with open(rule_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '|||' in line:
                    src, tgt = line.split('|||', 1)
                    src = src.strip()
                    tgt = tgt.strip()
                    
                    # Check if source is a regex pattern (contains special chars)
                    if any(char in src for char in '()[]{}.*+?^$\\|'):
                        try:
                            # Compile as regex pattern
                            pattern = re.compile(src)
                            rules.append(('regex', pattern, tgt))
                        except re.error:
                            print(f"Warning: Invalid regex pattern '{src}' - skipping", file=sys.stderr)
                    else:
                        # Treat as normal string replacement
                        rules.append(('string', src, tgt))
        return rules

    def _create_break_pattern(self):
        consonants = r"က-အ"
        punctuation = r"၊|။"
        subscript = r"္"
        a_that = r"်"
        return re.compile(rf"((?<!{subscript})([{consonants}]|{punctuation})(?![{a_that}{subscript}]))")

    def syllable_break(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        result = self.break_pattern.sub(r'|\1', text)
        if result.startswith('|'):
            result = result[1:]
        return result.split('|')

    def _get_lm_score(self, history, word):
        """Updated to handle both dict and kenlm.Model"""
        if isinstance(self.lm, dict):
            # Original ARPA dict lookup
            for n in range(min(len(history), self.max_order-1), -1, -1):
                ngram = ' '.join(history[-n:] + [word]) if n > 0 else word
                if ngram in self.lm:
                    return self.lm[ngram]
            return self.unk_logprob
        else:
            # KenLM binary model
            context = ' '.join(history[-self.max_order+1:] + [word])
            return self.lm.score(context, bos=False, eos=False)

    def _get_syl_score(self, word):
        if not self.syl_freq:
            return 0.0
        score = 0.0
        for syl in word:
            score += math.log(self.syl_freq.get(syl, 1))
        return score / len(word)

    def _get_dict_score(self, word):
        return self.dict_weight if word in self.word_dict else 0.0

    def _post_edit(self, line):
        for rule_type, src, tgt in self.post_rules:
            if rule_type == 'regex':
                line = src.sub(tgt, line)
            else:
                line = line.replace(src, tgt)
        return line

    def _forward_mm(self, syllables):
        result = []
        i = 0
        while i < len(syllables):
            for j in range(min(self.max_word_len, len(syllables) - i), 0, -1):
                word = ''.join(syllables[i:i + j])
                if word in self.word_dict:
                    result.append((i, i + j, word))
                    i += j
                    break
            else:
                result.append((i, i + 1, syllables[i]))
                i += 1
        return result

    def _backward_mm(self, syllables):
        result = []
        i = len(syllables)
        while i > 0:
            for j in range(min(self.max_word_len, i), 0, -1):
                word = ''.join(syllables[i - j:i])
                if word in self.word_dict:
                    result.insert(0, (i - j, i, word))
                    i -= j
                    break
            else:
                result.insert(0, (i - 1, i, syllables[i - 1]))
                i -= 1
        return result

    def _get_bimm_segmentation(self, syllables):
        fmm = self._forward_mm(syllables)
        bmm = self._backward_mm(syllables)
        return fmm if len(fmm) <= len(bmm) else bmm

    def _visualize_dag(self, dag_edges, syllables, line_idx):
        dot_lines = ['digraph DAG {']
        dot_lines.append('  rankdir=LR;')
        for start, edges in dag_edges.items():
            for end, word, score, is_bimm in edges:
                label = f"{word} ({score:.1f}){'*' if is_bimm else ''}"
                dot_lines.append(f'  {start} -> {end} [label="{label}"];')
        dot_lines.append('}')
        dot_path = os.path.join(self.dag_output_dir, f'dag_line_{line_idx:04d}.dot')
        pdf_path = dot_path.replace('.dot', '.pdf')
        with open(dot_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dot_lines))
        subprocess.run(['dot', '-Tpdf', dot_path, '-o', pdf_path])

    def _remove_all_spaces(self, text):
        return text.replace(' ', '')

    def _remove_myanmar_spaces(self, text, preserve_digits=False):
        if preserve_digits:
            for pattern, replacement in PROTECT_SPACES:
                text = pattern.sub(replacement, text)

        prev = None
        while prev != text:
            prev = text
            text = RE_MM_LETTER_SPACE.sub(r'\1\2', text)

        if preserve_digits:
            text = text.replace('☃', ' ')

        return text

    def _preprocess_text(self, text):
        if not self.space_remove_mode:
            return text
        if self.space_remove_mode == 'all':
            return self._remove_all_spaces(text)
        elif self.space_remove_mode == 'my':
            return self._remove_myanmar_spaces(text, preserve_digits=False)
        elif self.space_remove_mode == 'my_not_num':
            return self._remove_myanmar_spaces(text, preserve_digits=True)
        return text

    def segment(self, text, line_idx=0):
        text = self._preprocess_text(text)
        syllables = self.syllable_break(text)
        n = len(syllables)
        dag = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, min(i + self.max_word_len + 1, n + 1)):
                word = ''.join(syllables[i:j])
                if word in self.word_dict or j - i == 1:
                    dag[i].append((j, word, False))

        # Add Bi-MM fallback path
        if self.use_bimm_fallback:
            bimmpath = self._get_bimm_segmentation(syllables)
            for start, end, word in bimmpath:
                dag[start].append((end, word, True))

        # Viterbi decoding
        scores = [-float('inf')] * (n + 1)
        paths = [None] * (n + 1)
        histories = [[] for _ in range(n + 1)]
        scores[0] = 0

        for i in range(n):
            for j, word, is_bimm in dag[i]:
                dict_score = self._get_dict_score(word)
                syl_score = self._get_syl_score(word)
                lm_score = self._get_lm_score(histories[i], word) if self.lm else 0.0
                total = dict_score + syl_score + lm_score
                if is_bimm:
                    total += self.bimm_boost
                if scores[j] < scores[i] + total:
                    scores[j] = scores[i] + total
                    paths[j] = (i, word)
                    histories[j] = histories[i] + [word]

        if self.visualize_dag:
            viz_dag = defaultdict(list)
            for i in range(n):
                for j, word, is_bimm in dag[i]:
                    dict_score = self._get_dict_score(word)
                    syl_score = self._get_syl_score(word)
                    lm_score = self._get_lm_score([], word) if self.lm else 0.0
                    score = dict_score + syl_score + lm_score + (self.bimm_boost if is_bimm else 0)
                    viz_dag[i].append((j, word, score, is_bimm))
            self._visualize_dag(viz_dag, syllables, line_idx)

        result = []
        idx = n
        while idx > 0:
            prev, word = paths[idx]
            result.append(word)
            idx = prev

        segmented = ' '.join(reversed(result))
        segmented = re.sub(r'\s+', ' ', segmented.strip())  # to normalize spaces
        return self._post_edit(segmented) if self.post_rules else segmented


def main():
    parser = argparse.ArgumentParser(
        description="oppa_word, Hybrid DAG + BiMM + LM Myanmar Word Segmenter with optional Aho-Corasick support"
    )
    parser.add_argument('--input', '-i', required=True,
                        help="Input file with one sentence per line (UTF-8)")
    parser.add_argument('--output', '-o',
                        help="Optional output file path (default: stdout)")
    parser.add_argument('--dict', '-d', required=True,
                        help="Word dictionary file (one word per line)")
    parser.add_argument('--sylfreq', '-s',
                        help="Syllable frequency file (syllable<TAB>frequency, for scoring)")
    parser.add_argument('--arpa', '-a',
                        help="ARPA-format syllable-level language model (optional)")
    parser.add_argument('--postrule-file',
                        help="Optional post-processing rules (e.g., merging, corrections)")
    parser.add_argument('--max-order', type=int, default=5,
                        help="Max LM n-gram order (default: 5)")
    parser.add_argument('--dict-weight', type=float, default=10.0,
                        help="Dictionary path weight in scoring (default: 10.0)")
    parser.add_argument('--use-bimm-fallback', action='store_true',
                        help="Enable Bi-directional Maximum Matching as fallback")
    parser.add_argument('--bimm-boost', type=float, default=0.0,
                        help="Boost score added to Bi-MM fallback path (default: 0.0)")
    parser.add_argument('--visualize-dag', action='store_true',
                        help="Generate DAG visualization (PDF per sentence)")
    parser.add_argument('--dag-output-dir', default='dag_viz',
                        help="Directory to save DAG PDFs if --visualize-dag is used (default: 'dag_viz')")
    parser.add_argument('--space-remove-mode', choices=['all', 'my', 'my_not_num'],
                        help="Preprocessing mode to remove spaces: 'all', 'my' (Myanmar only), or 'my_not_num (Myanmar but not including Myanmar numbers'")
    parser.add_argument('--max-word-len', type=int, default=6,
                       help="Maximum word length in syllables (3-12, default:6)")

    args = parser.parse_args()

    # Validate max_word_len
    if not 3 <= args.max_word_len <= 12:
        parser.error("--max-word-len must be between 3 and 12")

    segmenter = HybridDAGSegmenter(
        dict_path=args.dict,
        syl_freq_path=args.sylfreq,
        arpa_lm_path=args.arpa,
        max_order=args.max_order,
        dict_weight=args.dict_weight,
        postrule_file=args.postrule_file,
        use_bimm_fallback=args.use_bimm_fallback,
        bimm_boost=args.bimm_boost,
        visualize_dag=args.visualize_dag,
        dag_output_dir=args.dag_output_dir,
        space_remove_mode=args.space_remove_mode,
        max_word_len=args.max_word_len
    )

    with open(args.input, encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    output_lines = []
    for idx, line in enumerate(lines):
        segmented = segmenter.segment(line, idx)
        output_lines.append(segmented)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(output_lines) + '\n')
    else:
        for line in output_lines:
            print(line)


if __name__ == '__main__':
    main()

