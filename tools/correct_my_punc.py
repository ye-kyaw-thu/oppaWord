#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse

def correct_segmentation(text):
    """Correct Myanmar text segmentation by adding space before ၊ and ။"""
    return re.sub(r'(\S)([၊။])', r'\1 \2', text)

def process_stream(input_stream, output_stream):
    """Process input stream and write to output stream"""
    for line in input_stream:
        corrected = correct_segmentation(line)
        output_stream.write(corrected)

def main():
    parser = argparse.ArgumentParser(
        description='Correct Myanmar text segmentation by adding space before ၊ and ။',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--input', 
                        help='Input file (default: stdin)',
                        type=str,
                        default=None)
    parser.add_argument('-o', '--output', 
                        help='Output file (default: stdout)',
                        type=str,
                        default=None)
    
    args = parser.parse_args()
    
    # Handle input
    if args.input:
        try:
            input_stream = open(args.input, 'r', encoding='utf-8')
        except IOError as e:
            print(f"Error opening input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        input_stream = sys.stdin
    
    # Handle output
    if args.output:
        try:
            output_stream = open(args.output, 'w', encoding='utf-8')
        except IOError as e:
            print(f"Error opening output file: {e}", file=sys.stderr)
            if args.input:
                input_stream.close()
            sys.exit(1)
    else:
        output_stream = sys.stdout
    
    # Process the text
    try:
        process_stream(input_stream, output_stream)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up file handles if they're not stdin/stdout
        if args.input and input_stream != sys.stdin:
            input_stream.close()
        if args.output and output_stream != sys.stdout:
            output_stream.close()

if __name__ == '__main__':
    main()

