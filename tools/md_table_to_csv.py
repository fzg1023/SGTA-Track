#!/usr/bin/env python3
"""Extract a Markdown table from a document and write it out as CSV.

Written for "把 SGTA_ABLATION_PLAN.md 里的表 3.1 存成 csv": the rows must come
from the document itself rather than being retyped, so this parses the file.

Section semantics: `--section 3.1` starts at the heading whose text begins with
`3.1` (so `3.1 表 B...` matches but `3.1.3 ...` does not) and ends at the next
heading of the SAME OR HIGHER level, so sub-subsections cannot leak in.

Usage:
  python tools/md_table_to_csv.py --md docs/SGTA_ABLATION_PLAN.md \
      --section 3.1 --out docs/table_3.1_component_ladder.csv
  python tools/md_table_to_csv.py --md docs/SGTA_ABLATION_PLAN.md --section 3.1 --list
  python tools/md_table_to_csv.py --md docs/X.md --section 3 --index 2 --keep-markup
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import List, Optional, Tuple

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
# `3.1` must not be followed by another digit, otherwise `3.1.3` would match.
_EMPHASIS = [("**", ""), ("__", ""), ("`", "")]


def heading(line: str) -> Optional[Tuple[int, str]]:
    match = _HEADING.match(line)
    if not match:
        return None
    return len(match.group(1)), match.group(2)


def split_cells(line: str) -> List[str]:
    """Cells of one `| a | b |` row, without the empty outer fields."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def is_separator(cells: List[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL.match(cell) for cell in cells)


def clean(text: str, keep_markup: bool) -> str:
    if keep_markup:
        return text
    out = text
    for token, replacement in _EMPHASIS:
        out = out.replace(token, replacement)
    return out.strip()


def find_section(lines: List[str], name: str) -> Tuple[int, int]:
    """(start, end) line indices [start, end) of the named section."""
    pattern = re.compile(r"^" + re.escape(name) + r"(?![\d.])")
    start = -1
    level = 0
    for index, line in enumerate(lines):
        info = heading(line)
        if not info:
            continue
        if start < 0:
            if pattern.match(info[1]):
                start, level = index, info[0]
        elif info[0] <= level:
            return start, index
    if start < 0:
        raise SystemExit(f"section {name!r} not found")
    return start, len(lines)


def find_tables(lines: List[str], start: int, end: int
                ) -> List[Tuple[int, List[str], List[List[str]]]]:
    """Every Markdown table in [start, end) as (line_no, header, rows)."""
    tables: List[Tuple[int, List[str], List[List[str]]]] = []
    index = start
    while index < end:
        line = lines[index]
        if not line.strip().startswith("|"):
            index += 1
            continue
        block_start = index
        block: List[str] = []
        while index < end and lines[index].strip().startswith("|"):
            block.append(lines[index])
            index += 1
        if len(block) < 2:
            continue
        header = split_cells(block[0])
        body = block[1:]
        if is_separator(split_cells(body[0])):
            body = body[1:]
        rows = [split_cells(row) for row in body]
        tables.append((block_start + 1, header, rows))
    return tables


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Markdown file to read")
    parser.add_argument("--section", required=True,
                        help="section number whose heading starts with it, e.g. 3.1")
    parser.add_argument("--index", type=int, default=1,
                        help="which table inside the section (1-based, default 1)")
    parser.add_argument("--out", default="", help="CSV path to write")
    parser.add_argument("--list", action="store_true",
                        help="only list the tables found in the section")
    parser.add_argument("--keep-markup", action="store_true",
                        help="keep **bold**/`code` markers instead of stripping them")
    parser.add_argument("--encoding", default="utf-8-sig",
                        help="output encoding (default utf-8-sig so Excel shows CJK)")
    args = parser.parse_args()

    with open(args.md, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    start, end = find_section(lines, args.section)
    tables = find_tables(lines, start, end)
    if not tables:
        raise SystemExit(f"no Markdown table found in section {args.section}")
    if args.list:
        print(f"section {args.section}: lines {start + 1}..{end} "
              f"({len(tables)} table(s))")
        for position, (line_no, header, rows) in enumerate(tables, start=1):
            print(f"  [{position}] line {line_no}: {len(header)} cols, "
                  f"{len(rows)} rows -> {header}")
        return 0

    position = args.index
    if not 1 <= position <= len(tables):
        raise SystemExit(f"--index {position} out of range "
                         f"(section has {len(tables)} table(s))")
    line_no, header, rows = tables[position - 1]

    width = len(header)
    cleaned_header = [clean(cell, args.keep_markup) for cell in header]
    cleaned_rows: List[List[str]] = []
    for offset, row in enumerate(rows, start=1):
        if len(row) != width:
            print(f"[WARN] line {line_no + offset + 1}: {len(row)} cells, "
                  f"expected {width}; padding", file=sys.stderr)
            row = (row + [""] * width)[:width]
        cleaned_rows.append([clean(cell, args.keep_markup) for cell in row])

    out = args.out
    if not out:
        base = os.path.splitext(os.path.basename(args.md))[0]
        out = f"{base}_table_{args.section}.csv"

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding=args.encoding) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(cleaned_header)
        writer.writerows(cleaned_rows)

    print(f"source : {args.md} line {line_no} (section {args.section}, "
          f"table {position}/{len(tables)})")
    print(f"wrote  : {out}  ({len(cleaned_rows)} rows + header, {width} cols)")
    print(f"columns: {cleaned_header}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
