#!/usr/bin/env python3
"""Deterministic identity hash for the live SUPPORTED task pool.

The hash is sha256 over the sorted (task_id, prompt) pairs of every supported
row (explicit dataset label `evaluation_backend in {"wcc", "ops6k"}` with a
non-empty prompt), truncated to 12 hex chars. Two pools hash equal iff they
contain the same tasks with the same prompts — prompt edits (e.g. a contract
migration) and membership changes (e.g. relabeling a dead row unsupported)
both change the hash.

THIS VALUE GOES INTO THE LEDGER: record it in the `task_pool_hash` column of
`research/results.tsv` (19-column schema, 2026-08-10) and the
`task_pool_hash` field of `research/live/master.json` for every run, so any
result row can be tied to the exact pool composition that produced it.
Rows with different task_pool_hash values are not comparable.

Membership uses the EXPLICIT dataset label, not
training.task_support.normalize_task_row, which recomputes support from
task_code and (as of 2026-08-10) would re-include rows explicitly relabeled
unsupported; see research/audits/2026-08-10-holdout-handoff.md.

Task ids: rows carry no task_id column in datasets/combined_kernelforge.jsonl,
so a stable id is derived per row: explicit `task_id`, else `kernel_id`, else
`ops6k_<sha1(task_code)[:12]>` (identical to tasks/build_task_pool.py), else
`prompt_<sha1(prompt)[:12]>`.

Usage:
    python scripts/task_pool_hash.py [--dataset PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "combined_kernelforge.jsonl"

SUPPORTED_BACKENDS = {"wcc", "ops6k"}


def stable_task_id(row: dict) -> str:
    """Derive a stable task id for a dataset row (see module docstring)."""
    explicit = str(row.get("task_id") or "").strip()
    if explicit:
        return explicit
    kernel_id = str(row.get("kernel_id") or "").strip()
    if kernel_id:
        return kernel_id
    task_code = str(row.get("task_code") or "").strip()
    if task_code:
        digest = hashlib.sha1(task_code.encode("utf-8")).hexdigest()[:12]
        return f"ops6k_{digest}"
    prompt = str(row.get("prompt") or "").strip()
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"prompt_{digest}"


def is_supported_row(row: dict) -> bool:
    """Supported per the EXPLICIT dataset label (see module docstring)."""
    return (
        row.get("evaluation_backend") in SUPPORTED_BACKENDS
        and bool(str(row.get("prompt") or "").strip())
    )


def load_rows(dataset_path: str | Path) -> list[dict]:
    path = Path(dataset_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_supported_rows(dataset_path: str | Path = DEFAULT_DATASET) -> list[dict]:
    return [row for row in load_rows(dataset_path) if is_supported_row(row)]


def compute_pool_hash(rows: list[dict]) -> str:
    """sha256 (12 hex chars) over the sorted (task_id, prompt) pairs of rows.

    Callers pass the supported pool (see load_supported_rows); the function
    hashes exactly the rows it is given.
    """
    pairs = sorted(
        (stable_task_id(row), str(row.get("prompt") or "")) for row in rows
    )
    h = hashlib.sha256()
    for task_id, prompt in pairs:
        # json array encoding makes the (id, prompt) boundary unambiguous.
        h.update(json.dumps([task_id, prompt], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    rows = load_supported_rows(args.dataset)
    ops6k = sum(1 for r in rows if r.get("evaluation_backend") == "ops6k")
    wcc = sum(1 for r in rows if r.get("evaluation_backend") == "wcc")
    print(f"task_pool_hash: {compute_pool_hash(rows)}")
    print(f"pool_size: {len(rows)} (ops6k={ops6k}, wcc={wcc})")
    print("Record task_pool_hash in research/results.tsv (task_pool_hash column) "
          "and research/live/master.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
