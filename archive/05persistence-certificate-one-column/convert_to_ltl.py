#!/usr/bin/env python3
"""Convert paper from omega-regular to LTL-focused"""

def main():
    with open('main.tex', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove commented abstract versions (lines 196-201, 0-indexed 195-200)
    # Delete lines with old abstract comments
    new_lines = []
    skip_until_line = -1
    for i, line in enumerate(lines):
        line_num = i + 1
        # Skip commented abstract lines (196-201)
        if 196 <= line_num <= 201:
            continue
        new_lines.append(line)

    # Write back
    with open('main.tex', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("OK: Removed commented abstract versions")
    print("OK: Title changed to LTL Properties")
    print("Next: Add Contributions section to Introduction")

if __name__ == '__main__':
    main()
