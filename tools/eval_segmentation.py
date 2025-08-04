#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Word Segmentation Evaluator with Error Analysis

Enhanced Word Segmentation Evaluator with Error Analysis
Written by Ye Kyaw Thu, LU Lab., Myanmar.
Last updated: 4 Aug 2025

Features:
- Precision, Recall, F1 metrics at word/boundary/vocab levels
- Top-K most frequent segmentation errors
- Detailed statistics

Usage:  
 python ./tools/eval_segmentation.py -r reference.txt -H  token.txt > eval_result.txt
"""

import argparse
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set

def read_lines(file_path: str) -> List[str]:
    """Read lines from file or stdin"""
    if file_path == '-':
        return [line.strip() for line in sys.stdin if line.strip()]
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_word_boundaries(text: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Extract word boundaries and words from text
    Returns: (list of (start, end) positions, list of words)
    """
    words = text.split()
    boundaries = []
    pos = 0
    for word in words:
        start = pos
        end = pos + len(word)
        boundaries.append((start, end))
        pos = end + 1  # +1 for the space
    return boundaries, words

def analyze_errors(ref_lines: List[str], hyp_lines: List[str], top_k: int = 10) -> Dict:
    """
    Enhanced error analyzer for Myanmar text segmentation
    """
    error_stats = {
        'over_segmentation': Counter(),
        'under_segmentation': Counter(),
        'incorrect_boundaries': Counter(),
        'total_errors': 0
    }

    # Myanmar-specific particle patterns
    PARTICLES = {'ပါ', 'တယ်', 'သည်', '၏', 'ကို', 'မှာ', 'နဲ့', 'လည်း'}
    
    for ref, hyp in zip(ref_lines, hyp_lines):
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        ref_ptr = 0
        hyp_ptr = 0
        
        while ref_ptr < len(ref_words) and hyp_ptr < len(hyp_words):
            ref_word = ref_words[ref_ptr]
            hyp_word = hyp_words[hyp_ptr]
            
            # Case 1: Exact match
            if ref_word == hyp_word:
                ref_ptr += 1
                hyp_ptr += 1
                continue
                
            error_stats['total_errors'] += 1
            
            # Case 2: Potential particle attachment error
            for particle in PARTICLES:
                if hyp_word.endswith(particle):
                    stem = hyp_word[:-len(particle)]
                    if stem in ref_word:
                        error_stats['incorrect_boundaries'][f"REF: '{ref_word}' → HYP: '{hyp_word}'"] += 1
                        ref_ptr += 1
                        hyp_ptr += 1
                        continue
                
            # Case 3: Over-segmentation
            combined_hyp = hyp_word
            end_hyp = hyp_ptr + 1
            while end_hyp < len(hyp_words):
                temp_combined = combined_hyp + hyp_words[end_hyp]
                if temp_combined in ref_word:
                    combined_hyp = temp_combined
                    end_hyp += 1
                else:
                    break
                    
            if combined_hyp == ref_word:
                error_key = f"REF: '{ref_word}' → HYP: '{'|'.join(hyp_words[hyp_ptr:end_hyp])}'"
                error_stats['over_segmentation'][error_key] += 1
                hyp_ptr = end_hyp
                ref_ptr += 1
                continue
                
            # Case 4: Under-segmentation
            combined_ref = ref_word
            end_ref = ref_ptr + 1
            while end_ref < len(ref_words):
                temp_combined = combined_ref + ref_words[end_ref]
                if temp_combined in hyp_word:
                    combined_ref = temp_combined
                    end_ref += 1
                else:
                    break
                    
            if combined_ref == hyp_word:
                error_key = f"REF: '{'|'.join(ref_words[ref_ptr:end_ref])}' → HYP: '{hyp_word}'"
                error_stats['under_segmentation'][error_key] += 1
                ref_ptr = end_ref
                hyp_ptr += 1
                continue
                
            # Case 5: Complex boundary error
            error_key = f"REF: '{ref_word}' → HYP: '{hyp_word}'"
            error_stats['incorrect_boundaries'][error_key] += 1
            ref_ptr += 1
            hyp_ptr += 1

    # Filter and return results
    return {
        'over_segmentation': dict(error_stats['over_segmentation'].most_common(top_k)),
        'under_segmentation': dict(error_stats['under_segmentation'].most_common(top_k)),
        'incorrect_boundaries': dict(error_stats['incorrect_boundaries'].most_common(top_k)),
        'total_errors': error_stats['total_errors']
    }

def calculate_metrics(ref_lines: List[str], hyp_lines: List[str]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for word segmentation
    Returns: dictionary of metrics
    """
    if len(ref_lines) != len(hyp_lines):
        print("Warning: Reference and hypothesis have different line counts", file=sys.stderr)

    total_ref_words = 0
    total_hyp_words = 0
    correct_words = 0
    boundary_correct = 0
    boundary_total = 0
    boundary_predicted = 0

    # For type-level statistics
    vocab_ref = defaultdict(int)
    vocab_hyp = defaultdict(int)
    vocab_correct = defaultdict(int)

    for ref, hyp in zip(ref_lines, hyp_lines):
        # Get word boundaries
        ref_boundaries, ref_words = get_word_boundaries(ref)
        hyp_boundaries, hyp_words = get_word_boundaries(hyp)

        # Update vocabulary counts
        for word in ref_words:
            vocab_ref[word] += 1
        for word in hyp_words:
            vocab_hyp[word] += 1

        # Convert boundaries to sets for comparison
        ref_bound_set = set(ref_boundaries)
        hyp_bound_set = set(hyp_boundaries)

        # Boundary-level stats
        boundary_correct += len(ref_bound_set & hyp_bound_set)
        boundary_total += len(ref_bound_set)
        boundary_predicted += len(hyp_bound_set)

        # Word-level stats (exact match)
        ref_pos = 0
        hyp_pos = 0
        ref_idx = 0
        hyp_idx = 0

        while ref_idx < len(ref_words) and hyp_idx < len(hyp_words):
            ref_word = ref_words[ref_idx]
            hyp_word = hyp_words[hyp_idx]

            if ref_word == hyp_word:
                correct_words += 1
                vocab_correct[ref_word] += 1
                ref_pos += len(ref_word) + 1
                hyp_pos += len(hyp_word) + 1
                ref_idx += 1
                hyp_idx += 1
            elif ref_pos < hyp_pos:
                ref_pos += len(ref_word) + 1
                ref_idx += 1
            else:
                hyp_pos += len(hyp_word) + 1
                hyp_idx += 1

        total_ref_words += len(ref_words)
        total_hyp_words += len(hyp_words)

    # Calculate metrics
    precision = correct_words / total_hyp_words if total_hyp_words > 0 else 0
    recall = correct_words / total_ref_words if total_ref_words > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    boundary_precision = boundary_correct / boundary_predicted if boundary_predicted > 0 else 0
    boundary_recall = boundary_correct / boundary_total if boundary_total > 0 else 0
    boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0

    # Vocabulary statistics
    vocab_precision = len(set(vocab_hyp) & set(vocab_ref)) / len(vocab_hyp) if len(vocab_hyp) > 0 else 0
    vocab_recall = len(set(vocab_hyp) & set(vocab_ref)) / len(vocab_ref) if len(vocab_ref) > 0 else 0
    vocab_f1 = 2 * (vocab_precision * vocab_recall) / (vocab_precision + vocab_recall) if (vocab_precision + vocab_recall) > 0 else 0

    return {
        'word_precision': precision,
        'word_recall': recall,
        'word_f1': f1,
        'boundary_precision': boundary_precision,
        'boundary_recall': boundary_recall,
        'boundary_f1': boundary_f1,
        'vocab_precision': vocab_precision,
        'vocab_recall': vocab_recall,
        'vocab_f1': vocab_f1,
        'total_ref_words': total_ref_words,
        'total_hyp_words': total_hyp_words,
        'correct_words': correct_words,
        'vocab_ref_size': len(vocab_ref),
        'vocab_hyp_size': len(vocab_hyp),
        'vocab_common': len(set(vocab_hyp) & set(vocab_ref))
    }

def print_metrics(metrics: Dict[str, float], error_stats: Dict = None, top_k: int = 10):
    """Print formatted evaluation metrics and error analysis"""
    print("\nWord Segmentation Evaluation Results")
    print("=" * 60)
    print(f"{'Metric':<25} {'Score':>10}")
    print("-" * 60)
    print(f"{'Word Precision':<25} {metrics['word_precision']:>10.4f}")
    print(f"{'Word Recall':<25} {metrics['word_recall']:>10.4f}")
    print(f"{'Word F1-score':<25} {metrics['word_f1']:>10.4f}")
    print("-" * 60)
    print(f"{'Boundary Precision':<25} {metrics['boundary_precision']:>10.4f}")
    print(f"{'Boundary Recall':<25} {metrics['boundary_recall']:>10.4f}")
    print(f"{'Boundary F1-score':<25} {metrics['boundary_f1']:>10.4f}")
    print("-" * 60)
    print(f"{'Vocab Precision':<25} {metrics['vocab_precision']:>10.4f}")
    print(f"{'Vocab Recall':<25} {metrics['vocab_recall']:>10.4f}")
    print(f"{'Vocab F1-score':<25} {metrics['vocab_f1']:>10.4f}")
    print("=" * 60)
    print("\nAdditional Statistics:")
    print(f"Reference words: {metrics['total_ref_words']}")
    print(f"Hypothesis words: {metrics['total_hyp_words']}")
    print(f"Correct words: {metrics['correct_words']}")
    print(f"Reference vocabulary size: {metrics['vocab_ref_size']}")
    print(f"Hypothesis vocabulary size: {metrics['vocab_hyp_size']}")
    print(f"Common vocabulary: {metrics['vocab_common']}")

    if error_stats:
        print("\nTop Segmentation Errors Analysis")
        print("=" * 60)
        print(f"Total errors: {error_stats['total_errors']}")
        
        print("\nMost Frequent Over-Segmentation Errors (System split where it shouldn't):")
        for error, count in error_stats['over_segmentation'].items():
            print(f"{count:>5} × {error}")
            
        print("\nMost Frequent Under-Segmentation Errors (System joined what should be separate):")
        for error, count in error_stats['under_segmentation'].items():
            print(f"{count:>5} × {error}")
            
        print("\nMost Frequent Complex Boundary Errors:")
        for error, count in error_stats['incorrect_boundaries'].items():
            print(f"{count:>5} × {error}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Word Segmentation Evaluator with Error Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-r', '--reference', required=True,
                      help='Reference (gold standard) file')
    parser.add_argument('-H', '--hypothesis',
                      help='Hypothesis (system output) file (use - for stdin)',
                      default='-')
    parser.add_argument('--top-k', type=int, default=10,
                      help='Show top K most frequent errors')
    parser.add_argument('--no-errors', action='store_true',
                      help='Skip error analysis to save time')
    
    args = parser.parse_args()

    # Read input files
    try:
        ref_lines = read_lines(args.reference)
        hyp_lines = read_lines(args.hypothesis)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate metrics
    metrics = calculate_metrics(ref_lines, hyp_lines)

    # Analyze errors if requested
    error_stats = None
    if not args.no_errors:
        error_stats = analyze_errors(ref_lines, hyp_lines, args.top_k)

    # Print results
    print_metrics(metrics, error_stats, args.top_k)

if __name__ == "__main__":
    main()

