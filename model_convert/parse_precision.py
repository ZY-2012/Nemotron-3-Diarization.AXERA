#!/usr/bin/env python3
"""Parse a Pulsar2 quant precision table and report the worst operators.

    python parse_precision.py <precision_analysis_table.txt> [--json out.json]
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def parse_table(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    rows, current = [], None
    for line in lines:
        if line.startswith("├") or line.startswith("└"):
            if current:
                rows.append(current)
            current = None
            continue
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.strip("│").split("│")]
        if len(cells) != 8:
            continue
        if current is None:
            current = cells
        else:
            for index in range(8):
                if cells[index]:
                    current[index] = (current[index] or "") + cells[index]
    if current:
        rows.append(current)

    parsed = []
    for row in rows:
        try:
            cosine = float(re.sub(r"[^0-9.eE+-]", "", row[6]))
        except ValueError:
            continue
        parsed.append(
            {
                "op": row[0],
                "type": row[1],
                "tensor": row[2],
                "shape": row[3],
                "dtype": row[4],
                "qdtype": row[5],
                "cosine": cosine,
                "mse": row[7],
            }
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table")
    parser.add_argument("--json", default=None)
    parser.add_argument("--worst", type=int, default=15)
    args = parser.parse_args()

    parsed = parse_table(Path(args.table))
    parsed.sort(key=lambda row: row["cosine"])
    print(f"operators: {len(parsed)}")
    if parsed:
        print(f"cosine min {parsed[0]['cosine']:.5f} mean {statistics.mean(r['cosine'] for r in parsed):.5f}")
        buckets = {"<0.5": 0, "0.5-0.9": 0, "0.9-0.99": 0, ">=0.99": 0}
        for row in parsed:
            c = row["cosine"]
            key = "<0.5" if c < 0.5 else "0.5-0.9" if c < 0.9 else "0.9-0.99" if c < 0.99 else ">=0.99"
            buckets[key] += 1
        print("buckets:", buckets)
        print(f"worst {args.worst}:")
        for row in parsed[: args.worst]:
            print(
                f"  {row['cosine']:.4f} mse={row['mse']:>12s} {row['type']:10s} "
                f"{row['op'][:46]:46s} {row['shape']:16s} {row['dtype']}/{row['qdtype']}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(parsed, indent=1))
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
