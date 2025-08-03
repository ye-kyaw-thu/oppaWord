#!/usr/bin/env python3
import re
import sys
import argparse

def process_myanmar_text(content, font_command):
    r"""Wrap Myanmar text with \burmesefont command"""
    return re.sub(r"([\u1000-\u109F]+)", fr"{{{font_command} \1}}", content)

def main():
    parser = argparse.ArgumentParser(
        description=r"Wrap Myanmar text in LaTeX with font command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="Input .tex file (or stdin if not specified)"
    )
    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Output .tex file (or stdout if not specified)"
    )
    parser.add_argument(
        '--font-command',
        default=r'\\burmesefont',
        help=r"LaTeX font command to wrap Myanmar text (e.g., '\myfont')"
    )
    
    args = parser.parse_args()
    
    content = args.input.read()
    processed = process_myanmar_text(content, args.font_command)
    args.output.write(processed)

    # Close files if they're not stdin/stdout
    if args.input != sys.stdin:
        args.input.close()
    if args.output != sys.stdout:
        args.output.close()

if __name__ == "__main__":
    main()
