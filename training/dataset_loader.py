"""Unified dataset loading for KernelForge training stages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

def _probe_hf_datasets():
    """Import HuggingFace `datasets`, guarding against the repo's datasets/ dir.

    With ROOT on sys.path (or an editable-install path hook), `import datasets`
    can resolve to the repo's datasets/ directory as a namespace package. That
    shadow has no `Dataset` attribute, so verify inside the try and fall back
    to MiniDataset instead of raising AttributeError at import time.
    """
    root = str(Path(__file__).resolve().parents[1])
    cwd = os.getcwd()
    orig_sys_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in ("", ".", cwd, root)]
    try:
        import datasets as hf_datasets  # noqa: E402

        hf_datasets.Dataset  # namespace-package shadow lacks this attribute
        return hf_datasets
    except Exception:  # pragma: no cover - optional local dependency
        # Drop a cached shadow module so later probes can retry cleanly.
        shadow = sys.modules.get("datasets")
        if shadow is not None and not hasattr(shadow, "Dataset"):
            sys.modules.pop("datasets", None)
        return None
    finally:
        sys.path = orig_sys_path


_hf_datasets = _probe_hf_datasets()
Dataset = _hf_datasets.Dataset if _hf_datasets is not None else Any

from training.curriculum import CurriculumManager

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from build_combined_dataset import (  # noqa: E402
    DEFAULT_DG_MANIFEST,
    build_combined_dataset,
    inject_into_curriculum,
    write_jsonl,
)
from training.task_support import filter_supported_tasks, normalize_task_row, summarize_tasks

DEFAULT_COMBINED_PATH = ROOT / "datasets" / "combined_kernelforge.jsonl"
DEFAULT_DG_SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"


class MiniDataset(list):
    """Very small Dataset-compatible fallback for preflight and local smoke checks."""

    @property
    def column_names(self) -> list[str]:
        if not self:
            return []
        return sorted({key for row in self for key in row.keys()})

    def shuffle(self, seed: int = 42):
        import random

        rows = list(self)
        random.Random(seed).shuffle(rows)
        return MiniDataset(rows)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _to_prompt_dataset(rows: list[dict[str, Any]]) -> Dataset:
    normalized = [normalize_task_row(r) for r in rows if r.get("prompt")]
    if _hf_datasets is None:
        return MiniDataset(normalized)
    return Dataset.from_list(normalized)


def _load_or_build_combined_rows(
    dg_manifest: str,
    ops6k_max: int | None,
    seed: int,
    combined_output: str,
) -> list[dict[str, Any]]:
    """Load the combined dataset cache; never rebuild it implicitly.

    Rebuilding overwrites datasets/combined_kernelforge.jsonl, destroying any
    manually curated rows (e.g. the vector_add_e2 task). Regeneration requires
    KERNELFORGE_ALLOW_DATASET_REBUILD=1; a cache that looks inconsistent with
    the request is served as-is with a LOUD warning instead.
    """
    combined_path = Path(combined_output)
    allow_rebuild = os.getenv("KERNELFORGE_ALLOW_DATASET_REBUILD", "0") == "1"

    rows: list[dict[str, Any]] = _read_jsonl(combined_path) if combined_path.exists() else []

    if rows:
        looks_inconsistent = False
        if ops6k_max is not None and int(ops6k_max) > 0:
            has_ops_tasks = any(str(row.get("task_code") or "").strip() for row in rows)
            looks_inconsistent = not has_ops_tasks
        if not looks_inconsistent:
            return rows
        if not allow_rebuild:
            print(
                f"WARNING: {combined_path} looks inconsistent with this request "
                f"(no task_code rows but ops6k_max={ops6k_max} was requested). "
                "Serving the existing cache UNCHANGED — it may be stale or built "
                "with different params. Set KERNELFORGE_ALLOW_DATASET_REBUILD=1 "
                "to regenerate (this OVERWRITES manually curated rows).",
                flush=True,
            )
            return rows
        print(
            f"KERNELFORGE_ALLOW_DATASET_REBUILD=1 — rebuilding inconsistent cache "
            f"{combined_path} (ops6k_max={ops6k_max}).",
            flush=True,
        )
    elif not allow_rebuild:
        raise RuntimeError(
            f"Combined dataset {combined_path} is missing or empty, and implicit "
            "rebuilds are disabled. Set KERNELFORGE_ALLOW_DATASET_REBUILD=1 to "
            "build it (this writes the file from the doubleGraph manifest + "
            "Ops-6K, overwriting any manual curation)."
        )

    rows = build_combined_dataset(
        dg_path=dg_manifest,
        ops6k_max=ops6k_max,
        seed=seed,
    )
    write_jsonl(rows, combined_path)
    return rows


def load_training_dataset(
    stage: str,
    dg_manifest: str = str(DEFAULT_DG_MANIFEST),
    ops6k_max: int | None = 1024,
    seed: int = 42,
    combined_output: str = str(DEFAULT_COMBINED_PATH),
    sft_path: str = str(DEFAULT_DG_SFT_PATH),
    curriculum_manager: CurriculumManager | None = None,
) -> Dataset | list[dict[str, Any]]:
    """Load dataset appropriate for each training stage.

    stage1:
        Prompt Dataset with easy Ops-6K samples + doubleGraph base kernels.
    stage2:
        SFT rows from datasets/doublegraph_sft.jsonl.
    stage3:
        Full combined rows; optionally injected into provided CurriculumManager.
    """
    stage_key = stage.strip().lower()

    if stage_key not in {"stage1", "stage2", "stage3"}:
        raise ValueError("stage must be one of: stage1, stage2, stage3")

    if stage_key == "stage2":
        sft_file = Path(sft_path)
        if not sft_file.exists():
            return []
        return _read_jsonl(sft_file)

    rows = _load_or_build_combined_rows(
        dg_manifest=dg_manifest,
        ops6k_max=ops6k_max,
        seed=seed,
        combined_output=combined_output,
    )
    supported_rows = filter_supported_tasks(rows)

    if stage_key == "stage1":
        stage1_rows = [
            row
            for row in supported_rows
            if int(row.get("difficulty", 1)) == 1
            or (row.get("data_source") == "doublegraph_a100" and int(row.get("difficulty", 1)) == 2)
        ]
        if len(stage1_rows) < 16:
            stage1_rows = list(supported_rows)
        if not stage1_rows:
            raise RuntimeError(
                "No Stage 1 rows have a live evaluator. "
                f"Dataset summary: {summarize_tasks(rows)}"
            )
        return _to_prompt_dataset(stage1_rows)

    if curriculum_manager is not None:
        inject_into_curriculum(curriculum_manager, supported_rows)

    if not supported_rows:
        raise RuntimeError(
            "No live-evaluable tasks are available for GRPO. "
            f"Dataset summary: {summarize_tasks(rows)}"
        )
    return supported_rows
