#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
smart_space_remover.py: Remove spaces intelligently for Myanmar text segmentation.
written by Ye Kyaw Thu, LU Lab., Myanmar.
last update: 25 July 2025

Modes:
  - all     : Remove all spaces
  - my      : Remove spaces only between Myanmar letters
  - my_not_num  : Like 'my' but preserve spacing near Myanmar numbers

Usage:
  $ python smart_space_remover.py --mode my_not_num --input input.txt --output output.txt
"""

import sys
import argparse
import re

# === Unicode Character Classes ===
MYANMAR_LETTER = r'[\u1000-\u109F\uAA60-\uAA7F]'
MYANMAR_DIGIT = r'[\u1040-\u1049]'

# === Regex Patterns ===
RE_MM_LETTER_SPACE = re.compile(rf'({MYANMAR_LETTER})\s+({MYANMAR_LETTER})')

# Match MyanmarDigit <space> MyanmarDigit
RE_MM_DIGIT_DIGIT = re.compile(rf'({MYANMAR_DIGIT})\s+({MYANMAR_DIGIT})')

# Match MyanmarDigit <space> MyanmarLetter
RE_MM_DIGIT_LETTER = re.compile(rf'({MYANMAR_DIGIT})\s+({MYANMAR_LETTER})')

# Match MyanmarLetter <space> MyanmarDigit
RE_MM_LETTER_DIGIT = re.compile(rf'({MYANMAR_LETTER})\s+({MYANMAR_DIGIT})')

# Protect standalone Myanmar digit tokens and spacing
PROTECT_SPACES = [
    (RE_MM_DIGIT_DIGIT, r'\1☃\2'),           # protect digit-digit
    (RE_MM_DIGIT_LETTER, r'\1☃\2'),          # protect digit-letter
    (RE_MM_LETTER_DIGIT, r'\1☃\2'),          # protect letter-digit
]

def remove_all_spaces(text):
    return text.replace(' ', '')

def remove_myanmar_spaces(text, preserve_digits=False):
    if preserve_digits:
        # Step 1: Protect spaces between digits and letters
        for pattern, replacement in PROTECT_SPACES:
            text = pattern.sub(replacement, text)

    # Step 2: Remove space between Myanmar letters
    prev = None
    while prev != text:
        prev = text
        text = RE_MM_LETTER_SPACE.sub(r'\1\2', text)

    if preserve_digits:
        # Step 3: Restore protected spaces
        text = text.replace('☃', ' ')

    return text

def process_lines(lines, mode):
    for line in lines:
        line = line.rstrip('\n')
        if mode == 'all':
            yield remove_all_spaces(line)
        elif mode == 'my':
            yield remove_myanmar_spaces(line, preserve_digits=False)
        elif mode == 'my_not_num':
            yield remove_myanmar_spaces(line, preserve_digits=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description="Smart Myanmar space remover")
    parser.add_argument('--mode', choices=['all', 'my', 'my_not_num'], required=True,
                        help="Mode: 'all', 'my', or 'my+num'")
    parser.add_argument('--input', type=str, help="Input file (default: stdin)")
    parser.add_argument('--output', type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    input_stream = open(args.input, 'r', encoding='utf-8') if args.input else sys.stdin
    output_stream = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout

    try:
        for processed in process_lines(input_stream, args.mode):
            output_stream.write(processed + '\n')
    finally:
        if args.input:
            input_stream.close()
        if args.output:
            output_stream.close()

if __name__ == '__main__':
    main()

