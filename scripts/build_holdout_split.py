#!/usr/bin/env python3
"""Build a deterministic holdout eval split over the supported task pool.

Fixes the 2026-08-03 audit finding that evaluation/eval_model._load_eval_tasks
served the LAST 50 rows of the training set ("KNOWN CONTAMINATION" warning) —
eval and training overlapped completely. Owner decision 2026-08-10: build a
real holdout now.

Split rule (documented, deterministic, no RNG):
- Universe: the WCC+ops6k union — every row of the source dataset whose
  EXPLICIT `evaluation_backend` label is "wcc" or "ops6k" with a non-empty
  prompt. Chosen because the supported ops6k pool alone is only 14 rows
  (two 3+-tensor rows were relabeled unsupported on 2026-08-10); the union
  (18 rows) is the largest supported universe that exists locally — both
  backends have live evaluators, and evaluation/eval_model.py already scores
  both. Explicit labels are used, not normalize_task_row, which recomputes
  support from task_code and would re-include the relabeled dead rows.
- The CAMPAIGN-018 spike-pool tasks (vector_add_e2, F.elu, F.softplus) are
  pinned to the train side — they are the designated EXP-018c-p0 probe /
  018c-rerun tasks, and a holdout containing them would be contaminated by
  the very first probe.
- Holdout size: max(3, round(0.20 * universe)).
- Stratified by evaluation_backend (largest-remainder allocation, each
  backend keeps >=1 row on BOTH sides) so neither side loses a whole
  evaluator family.
- Within each backend, ELIGIBLE (non-pinned) rows are ranked by
  sha256("kernelforge-holdout-v2:" + task_id) and the lowest digests go to
  holdout. Membership therefore depends only on task identity — not on file
  order, insertion time, or an RNG seed.

Outputs (atomic writes):
- datasets/holdout_eval.jsonl      — full holdout rows + injected `task_id`
- datasets/holdout_manifest.json   — rule, source hash, both sides' task ids
                                     and pool hashes (scripts/task_pool_hash.py)

Non-overlap between holdout and train is ASSERTED in code on both task_id and
prompt. Training-side exclusion of these task ids lives in
training/dataset_loader.py — diff routed via
research/audits/2026-08-10-holdout-handoff.md (file owned by another worker).

Usage:
    python scripts/build_holdout_split.py [--dataset PATH] [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task_pool_hash import (  # noqa: E402
    DEFAULT_DATASET,
    compute_pool_hash,
    load_supported_rows,
    stable_task_id,
)

HOLDOUT_PATH = ROOT / "datasets" / "holdout_eval.jsonl"
MANIFEST_PATH = ROOT / "datasets" / "holdout_manifest.json"
SALT = "kernelforge-holdout-v2"
HOLDOUT_FRACTION = 0.20
HOLDOUT_MIN = 3

# CAMPAIGN-018 spike pool (EXP-018a "Concrete 3-task spike pool"): these tasks
# are the designated probe/training tasks for EXP-018c-p0 and the 018c rerun,
# so they are pinned to the TRAIN side — a holdout containing them would be
# contaminated by the very first probe. Stable ids per scripts/task_pool_hash.
PINNED_TRAIN_TASK_IDS = {
    "vector_add_e2",        # torch.add   (hand-crafted E2 sanity floor)
    "ops6k_4343751ebe34",   # F.elu       (E1)
    "ops6k_6645fc3a14d1",   # F.softplus  (E1)
}

RULE_TEXT = (
    "Universe: rows of the source dataset with explicit evaluation_backend in "
    "{wcc, ops6k} and a non-empty prompt (WCC+ops6k union; largest locally "
    "available supported universe). The CAMPAIGN-018 spike-pool tasks "
    "(vector_add_e2, F.elu, F.softplus) are pinned to the train side and are "
    "never holdout-eligible. Holdout size max(3, round(0.20*N)) over the full "
    "universe N, stratified by evaluation_backend via largest-remainder with "
    ">=1 row per backend on both sides; within a backend the ELIGIBLE rows "
    "with the lowest sha256('kernelforge-holdout-v2:' + task_id) digests are "
    "held out."
)


def _rank_key(task_id: str) -> str:
    return hashlib.sha256(f"{SALT}:{task_id}".encode("utf-8")).hexdigest()


def _allocate(counts: dict[str, int], k_total: int) -> dict[str, int]:
    """Largest-remainder allocation of k_total across backends.

    Guarantees 1 <= k_b <= n_b - 1 for every backend (both sides keep at
    least one row per backend).
    """
    n = sum(counts.values())
    if k_total >= n:
        raise SystemExit(
            f"Holdout size {k_total} would consume the whole universe ({n} rows)."
        )
    if any(c < 2 for c in counts.values()):
        raise SystemExit(
            f"Every backend needs >=2 rows to appear on both sides; got {counts}."
        )
    quotas = {b: k_total * c / n for b, c in counts.items()}
    alloc = {b: int(q) for b, q in quotas.items()}
    # largest remainders first
    for b in sorted(quotas, key=lambda b: quotas[b] - alloc[b], reverse=True):
        if sum(alloc.values()) == k_total:
            break
        alloc[b] += 1
    # clamp into [1, n_b - 1], rebalancing the difference
    for b in sorted(alloc):
        alloc[b] = max(1, min(alloc[b], counts[b] - 1))
    diff = k_total - sum(alloc.values())
    for b in sorted(alloc, key=lambda b: counts[b] - alloc[b], reverse=True):
        while diff > 0 and alloc[b] < counts[b] - 1:
            alloc[b] += 1
            diff -= 1
        while diff < 0 and alloc[b] > 1:
            alloc[b] -= 1
            diff += 1
    assert sum(alloc.values()) == k_total, (alloc, k_total)
    return alloc


def build_split(dataset_path: Path) -> dict:
    rows = load_supported_rows(dataset_path)
    for row in rows:
        row["task_id"] = stable_task_id(row)
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate stable task ids in supported universe.")

    pinned = [r for r in rows if r["task_id"] in PINNED_TRAIN_TASK_IDS]
    missing_pins = PINNED_TRAIN_TASK_IDS - {r["task_id"] for r in pinned}
    if missing_pins:
        raise SystemExit(
            f"Pinned spike-pool tasks not found in supported universe: {missing_pins}"
        )
    eligible = [r for r in rows if r["task_id"] not in PINNED_TRAIN_TASK_IDS]

    by_backend: dict[str, list[dict]] = {}
    for row in eligible:
        by_backend.setdefault(row["evaluation_backend"], []).append(row)

    n = len(rows)
    k_total = max(HOLDOUT_MIN, round(HOLDOUT_FRACTION * n))
    alloc = _allocate({b: len(v) for b, v in by_backend.items()}, k_total)

    holdout: list[dict] = []
    train: list[dict] = list(pinned)
    for backend in sorted(by_backend):
        ranked = sorted(by_backend[backend], key=lambda r: _rank_key(r["task_id"]))
        holdout.extend(ranked[: alloc[backend]])
        train.extend(ranked[alloc[backend]:])

    holdout.sort(key=lambda r: r["task_id"])
    train.sort(key=lambda r: r["task_id"])

    # Non-overlap is a hard invariant, asserted on both identity axes.
    holdout_ids = {r["task_id"] for r in holdout}
    train_ids = {r["task_id"] for r in train}
    assert not (holdout_ids & train_ids), "holdout/train task_id overlap"
    holdout_prompts = {r["prompt"] for r in holdout}
    train_prompts = {r["prompt"] for r in train}
    assert not (holdout_prompts & train_prompts), "holdout/train prompt overlap"
    assert len(holdout) + len(train) == n

    source_sha = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    manifest = {
        "version": SALT,
        "created": "2026-08-10",
        "rule": RULE_TEXT,
        "source_dataset": str(Path(dataset_path).relative_to(ROOT))
        if str(dataset_path).startswith(str(ROOT))
        else str(dataset_path),
        "source_dataset_sha256": source_sha,
        "universe_size": n,
        "holdout": {
            "size": len(holdout),
            "by_backend": {b: sum(1 for r in holdout if r["evaluation_backend"] == b) for b in sorted(by_backend)},
            "task_pool_hash": compute_pool_hash(holdout),
            "task_ids": sorted(holdout_ids),
        },
        "train": {
            "size": len(train),
            "by_backend": {b: sum(1 for r in train if r["evaluation_backend"] == b) for b in sorted(by_backend)},
            "task_pool_hash": compute_pool_hash(train),
            "task_ids": sorted(train_ids),
        },
    }
    return {"holdout": holdout, "train": train, "manifest": manifest}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing split matches a fresh deterministic build",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    result = build_split(args.dataset)
    manifest = result["manifest"]

    if args.check:
        if not MANIFEST_PATH.exists():
            print("ERROR: no existing manifest to check against.")
            return 1
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for side in ("holdout", "train"):
            if existing.get(side, {}).get("task_ids") != manifest[side]["task_ids"]:
                print(f"MISMATCH: {side} task ids differ from a fresh build.")
                return 1
        print("Split check OK — existing manifest matches a fresh deterministic build.")
        return 0

    _write_atomic(
        HOLDOUT_PATH,
        "\n".join(json.dumps(r, ensure_ascii=False) for r in result["holdout"]) + "\n",
    )
    _write_atomic(MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"holdout: {manifest['holdout']['size']} rows {manifest['holdout']['by_backend']} "
          f"pool_hash={manifest['holdout']['task_pool_hash']} -> {HOLDOUT_PATH}")
    print(f"train:   {manifest['train']['size']} rows {manifest['train']['by_backend']} "
          f"pool_hash={manifest['train']['task_pool_hash']}")
    print(f"manifest -> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
