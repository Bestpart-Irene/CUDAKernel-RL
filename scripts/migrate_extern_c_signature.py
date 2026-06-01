#!/usr/bin/env python3
"""EXP-018a Day 3: backfill extern_c_signature into combined_kernelforge.jsonl.

Inspects each ops_local_fallback row's task_code by exec'ing it, counts
tensor inputs from get_inputs(), and assigns:
    1 tensor -> {"class": "E1"}
    2 tensors -> {"class": "E2"}
    other -> field omitted; row stays "unsupported by harness"

Idempotent — re-running just overwrites in place. Run on a host with
torch installed (cluster or local with venv).

Usage:
    python scripts/migrate_extern_c_signature.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "combined_kernelforge.jsonl"


def infer_signature(task_code: str) -> dict | None:
    try:
        import torch
    except ImportError:
        print("WARNING: torch not available; falling back to source-pattern heuristic", file=sys.stderr)
        return _heuristic_signature(task_code)

    try:
        ns: dict = {}
        exec(task_code, ns)
        get_inputs = ns.get("get_inputs")
        if get_inputs is None:
            return None
        sample = get_inputs()
        if not isinstance(sample, (list, tuple)):
            sample = [sample]
        tensor_count = sum(1 for x in sample if isinstance(x, torch.Tensor))
        if tensor_count == 1:
            return {"class": "E1"}
        if tensor_count == 2:
            return {"class": "E2"}
        return None
    except Exception as exc:
        print(f"  WARN: exec failed ({type(exc).__name__}: {str(exc)[:80]}); using heuristic", file=sys.stderr)
        return _heuristic_signature(task_code)


def _heuristic_signature(task_code: str) -> dict | None:
    """Best-effort static fallback when torch is unavailable.

    Counts `torch.randn(` and similar tensor-construction calls inside
    get_inputs(). Adequate for the simple ops_local_fallback rows
    which use straightforward `return [torch.randn(...)]` shapes.
    """
    import re
    body_match = re.search(r"def get_inputs\(\):(.*?)(?=\ndef |\Z)", task_code, re.S)
    if not body_match:
        return None
    body = body_match.group(1)
    tensor_constructors = len(re.findall(
        r"\btorch\.(?:randn|rand|zeros|ones|empty|arange|linspace|tensor)\s*\(",
        body,
    ))
    if tensor_constructors == 1:
        return {"class": "E1"}
    if tensor_constructors == 2:
        return {"class": "E2"}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    counts = {"E1": 0, "E2": 0, "skipped": 0, "unchanged": 0}

    for row in rows:
        if row.get("data_source") != "ops_local_fallback":
            counts["unchanged"] += 1
            continue
        tc = row.get("task_code", "")
        if not tc:
            counts["skipped"] += 1
            continue
        sig = infer_signature(tc)
        if sig is None:
            counts["skipped"] += 1
            row["extern_c_signature"] = None  # explicit so future readers see "tried but inconclusive"
            continue
        row["extern_c_signature"] = sig
        counts[sig["class"]] += 1

    print(f"Total rows: {len(rows)}")
    print(f"  E1 (unary elementwise): {counts['E1']}")
    print(f"  E2 (binary elementwise): {counts['E2']}")
    print(f"  Skipped (cannot infer): {counts['skipped']}")
    print(f"  Unchanged (non-ops_local_fallback): {counts['unchanged']}")

    if args.dry_run:
        print("\n[dry-run] not writing")
        return 0

    out_lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    args.dataset.write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {len(rows)} rows back to {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
