#!/usr/bin/env python3
"""EXP-018d: migrate live ops6k dataset prompts to the extern "C" contract.

The EXP-018a evaluator rejects `torch/extension.h` / PYBIND11_MODULE at the
source scan, yet 15/16 live ops6k rows in `datasets/combined_kernelforge.jsonl`
still instruct the model to use exactly that contract. This script rewrites
ONLY the `prompt` column of `evaluation_backend == "ops6k"` rows whose prompt
still carries the legacy demands, using the same prompt builder + signature
inference the rest of the pipeline now uses (`_build_cuda_prompt`).

Safety: creates `<dataset>.bak`, writes atomically (tmp + os.replace), and is
idempotent (a second run migrates 0 rows). No other columns are touched.

Usage:
    python scripts/migrate_ops6k_prompts_extern_c.py [--dataset PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.cuda_agent_integration import _build_cuda_prompt  # noqa: E402
from training.task_support import infer_signature_class  # noqa: E402

# Match the legacy DEMAND lines, not any mention: the new contract's own
# "Do NOT include `<torch/extension.h>`" prohibition contains the same
# substring, so a bare "torch/extension.h" marker would break idempotency.
FORBIDDEN_MARKERS = (
    "include `#include <torch/extension.h>`",
    "PYBIND11_MODULE",
)
DEFAULT_DATASET = ROOT / "datasets" / "combined_kernelforge.jsonl"


def _row_id(row: dict) -> str:
    return str(row.get("kernel_id") or row.get("task_id") or row.get("ops") or "?")


def _needs_migration(row: dict) -> bool:
    if row.get("evaluation_backend") != "ops6k":
        return False
    prompt = row.get("prompt") or ""
    return any(marker in prompt for marker in FORBIDDEN_MARKERS)


def migrate_rows(rows: list[dict]) -> dict:
    """Rewrite poisoned ops6k prompts in place; returns migration stats."""
    stats: dict = {
        "total": len(rows),
        "migrated": 0,
        "uninferable": [],
        "skipped_no_task_code": [],
    }
    for row in rows:
        if not _needs_migration(row):
            continue
        task_code = row.get("task_code") or ""
        if not task_code:
            stats["skipped_no_task_code"].append(_row_id(row))
            continue
        if infer_signature_class(task_code) is None:
            # The evaluator will also fail to infer a signature for this row
            # ("Could not infer extern_c_signature") — the prompt is migrated
            # anyway (pybind demands removed) but the row needs curation.
            stats["uninferable"].append(_row_id(row))
        new_prompt = _build_cuda_prompt(
            {
                "code": task_code,
                "ops": row.get("ops"),
                "data_source": row.get("data_source", "unknown"),
            }
        )
        if not new_prompt:
            stats["skipped_no_task_code"].append(_row_id(row))
            continue
        row["prompt"] = new_prompt
        stats["migrated"] += 1
    return stats


def migrate_file(path: Path, dry_run: bool = False) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stats = migrate_rows(rows)
    if stats["migrated"] and not dry_run:
        shutil.copy2(path, str(path) + ".bak")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    stats = migrate_file(args.dataset, dry_run=args.dry_run)
    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}{stats['migrated']}/{stats['total']} rows migrated to the extern-C contract")
    if stats["uninferable"]:
        print(
            "WARNING: signature class uninferable (evaluator will reject these "
            f"rows too — curate or drop them): {stats['uninferable']}"
        )
    if stats["skipped_no_task_code"]:
        print(f"WARNING: skipped rows without task_code: {stats['skipped_no_task_code']}")
    if stats["migrated"] and not args.dry_run:
        print(f"Backup written to {args.dataset}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
