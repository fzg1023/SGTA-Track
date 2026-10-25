"""Deduplicate eval_history.csv rows by ckpt_tag, keeping the LAST occurrence.

Usage: python tools/dedupe_history.py <path_to_eval_history.csv>
"""
import csv
import sys


def main():
    path = sys.argv[1]
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    seen = {}
    for row in body:
        if row:
            seen[row[0]] = row
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for tag in sorted(seen):
            w.writerow(seen[tag])
    print("deduped: %d rows -> %d rows" % (len(body), len(seen)))


if __name__ == "__main__":
    main()
