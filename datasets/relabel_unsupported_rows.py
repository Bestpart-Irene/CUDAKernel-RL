#!/usr/bin/env python3
"""Relabel permanently-dead ops6k rows as unsupported (owner decision 2026-08-10).

EXP-018d left two live ops6k rows with 3+ tensor inputs. Neither the contract
generator (training/task_support.infer_signature_class) nor the evaluator
(eval_service.eval_core._infer_extern_c_signature) can produce an extern-C
signature for them (only E1/E2 exist), so the evaluator permanently fails with
"Could not infer extern_c_signature" — reward pinned at -1 for tasks labeled
"supported". Owner decision 2026-08-10: relabel `unsupported`; an E3 signature
class is deferred. Targets (matched by exact ops list):

- ['F.softmax', 'F.scaled_dot_product_attention']
- ['torch.stack', 'torch.masked_select']

The dataset-level label (`evaluation_backend` / `supports_evaluation` /
`support_reason`) is the same mechanism datasets/build_combined_dataset.py uses
to mark every other unsupported row, and it is what
openenv_env/task_pool.TaskPool.load's fallback filters on. NOTE:
training/task_support.normalize_task_row currently RECOMPUTES the backend from
task_code and would flip these rows back to "ops6k"; the sticky-label fix for
that path is routed via research/audits/2026-08-10-holdout-handoff.md (that
file is owned by another worker).

Safety: atomic write (tmp + os.replace), no .bak file, idempotent — a second
run relabels 0 rows. Only the three label fields are touched.

Usage:
    python datasets/relabel_unsupported_rows.py [--dataset PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "datasets" / "combined_kernelforge.jsonl"

# Exact ops lists of the two rows the 2026-08-10 owner decision covers.
DEAD_ROW_OPS = (
    ["F.softmax", "F.scaled_dot_product_attention"],
    ["torch.stack", "torch.masked_select"],
)

UNSUPPORTED_REASON = (
    "3+ tensor inputs: no inferable extern-C signature (only E1/E2 exist; "
    "E3 deferred). Owner decision 2026-08-10: unsupported until an E3 "
    "signature class ships. See research/experiments/"
    "EXP-018d-contract-migration.md."
)


def _is_dead_row(row: dict) -> bool:
    return row.get("evaluation_backend") == "ops6k" and row.get("ops") in list(
        DEAD_ROW_OPS
    )


def _relabel_row(row: dict) -> dict:
    """Return the relabeled copy of a dead row (sanity-checked)."""
    # Sanity: the row must actually be signature-uninferable. Refuse to
    # relabel a row the contract generator can handle.
    from training.task_support import infer_signature_class

    sig = infer_signature_class(row.get("task_code") or "")
    if sig is not None:
        raise SystemExit(
            f"REFUSING relabel: row ops={row.get('ops')} infers signature "
            f"class {sig}; it is not a dead row."
        )
    out = dict(row)
    out["evaluation_backend"] = "unsupported"
    out["supports_evaluation"] = False
    out["support_reason"] = UNSUPPORTED_REASON
    return out


def relabel_file(path: Path, dry_run: bool = False) -> dict:
    """Relabel dead rows; untouched lines are preserved byte-for-byte."""
    stats: dict = {"total": 0, "relabeled": 0, "relabeled_ops": []}
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["total"] += 1
        row = json.loads(line)
        if _is_dead_row(row):
            new_row = _relabel_row(row)
            out_lines.append(json.dumps(new_row, ensure_ascii=False))
            stats["relabeled"] += 1
            stats["relabeled_ops"].append(row.get("ops"))
        else:
            out_lines.append(line)
    if stats["relabeled"] and not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        tmp.replace(path)  # atomic; no .bak by owner decision
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    stats = relabel_file(args.dataset, dry_run=args.dry_run)
    mode = "DRY RUN — " if args.dry_run else ""
    print(
        f"{mode}{stats['relabeled']}/{stats['total']} rows relabeled unsupported "
        f"(ops: {stats['relabeled_ops']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
