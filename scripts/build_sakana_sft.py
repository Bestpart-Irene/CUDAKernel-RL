"""Build a contract-aligned SFT dataset from SakanaAI/AI-CUDA-Engineer-Archive.

Goal: produce `datasets/sakana_sft.jsonl` in HF messages format whose
assistant messages use the ops6k `torch::Tensor run_kernel(...)`
return-value contract — replacing the doubleGraph WCC-style void
contract that biased EXP-002's SFT and broke EXP-003 verification.

Pipeline:
  1. Load SakanaAI/AI-CUDA-Engineer-Archive (Level 1 split, ~12k rows).
  2. Filter Correct=True, Max_Diff small, CUDA_Speedup_Native > 0.5
     (kernel runs and isn't catastrophically slow).
  3. Rename `kernel_function` → `run_kernel` in the CUDA source so the
     pybind name matches `eval_service.eval_core` expectations.
  4. Build user message in the same shape as
     `training.cuda_agent_integration._build_cuda_prompt` (so the model
     sees the same prompt distribution at Stage 1 GRPO inference time).
  5. Write HF messages JSONL with system/user/assistant turns. Limit to
     MAX_SAMPLES rows (default 200).

Run on CPU; no GPU needed. Downloads ~several GB of HF cache.

Usage:
  python -m scripts.build_sakana_sft  # writes datasets/sakana_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "datasets" / "sakana_sft.jsonl"


def _load_hf_datasets():
    cwd = os.getcwd()
    orig_sys_path = list(sys.path)
    try:
        shadow_paths = {"", ".", cwd, str(ROOT), str(ROOT / "datasets")}
        sys.path = [p for p in sys.path if p not in shadow_paths]
        import datasets as hf_datasets  # noqa: WPS433
        return hf_datasets
    finally:
        sys.path = orig_sys_path


def rename_kernel_function_to_run_kernel(code: str) -> str:
    """Rewrite `kernel_function` (Sakana's pybind name) → `run_kernel` (ours).

    Sakana exports `m.def("kernel_function", &kernel_function)`; eval_core
    looks for `run_kernel`. Rewrite occurrences of the bare identifier and
    the pybind string literal but leave anything else (e.g. comments,
    unrelated identifiers) alone.
    """
    code = re.sub(r"\bkernel_function\b", "run_kernel", code)
    return code


def build_messages(row: dict, target_gpu: str = "A100", target_arch: str = "sm_80") -> dict:
    """Compose one HF-messages SFT example from a Sakana row."""
    pytorch_ref = str(row.get("PyTorch_Code_Module") or "").strip()
    cuda_code = rename_kernel_function_to_run_kernel(str(row.get("CUDA_Code") or "").strip())
    op_name = str(row.get("Op_Name") or "").strip()

    system_msg = (
        "You are a CUDA kernel expert. Implement the requested PyTorch operator as a "
        "single PyTorch CUDA extension source file, exposing `run_kernel` via "
        "`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`. The function must accept the "
        "tensors returned by `get_inputs()` and return outputs matching `Model(*inputs)`."
    )
    user_msg = (
        f"Write a high-performance CUDA kernel implementation for NVIDIA {target_gpu} ({target_arch}) "
        f"that matches the semantics of the PyTorch reference below.\n\n"
        f"Reference op: {op_name}\n\n"
        "Reference implementation:\n"
        "```python\n"
        f"{pytorch_ref}\n"
        "```\n\n"
        "Return a single CUDA/C++ source file only. "
        "The source must include `#include <torch/extension.h>` and define `run_kernel`, "
        "exported via `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`. "
        "If extra nvcc flags are required, add a `// CU_FLAGS:` comment."
    )
    assistant_msg = "```cpp\n" + cuda_code + "\n```"
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def filter_row(row: dict, min_speedup: float, max_diff: float) -> bool:
    if not row.get("Correct"):
        return False
    speedup = row.get("CUDA_Speedup_Native")
    if speedup is None or float(speedup) < min_speedup:
        return False
    diff = row.get("Max_Diff")
    if diff is not None and float(diff) > max_diff:
        return False
    pytorch_ref = row.get("PyTorch_Code_Module") or ""
    cuda_code = row.get("CUDA_Code") or ""
    if not pytorch_ref.strip() or not cuda_code.strip():
        return False
    if "run_kernel" in cuda_code or "kernel_function" in cuda_code:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum SFT rows to emit (default 200).")
    parser.add_argument("--min-speedup", type=float, default=0.5,
                        help="Minimum CUDA_Speedup_Native to keep a row.")
    parser.add_argument("--max-diff", type=float, default=1e-2,
                        help="Maximum Max_Diff to keep a row (correctness margin).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=str, default="level_1",
                        help="Comma-separated split names (e.g. 'level_1' or 'level_1,level_2').")
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    hf_datasets = _load_hf_datasets()
    print(f"Loading SakanaAI/AI-CUDA-Engineer-Archive splits={args.levels}...")
    splits = [s.strip() for s in args.levels.split(",") if s.strip()]
    parts = []
    for split in splits:
        d = hf_datasets.load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=split)
        print(f"  {split}: {len(d)} rows")
        parts.append(d)
    ds = hf_datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    print(f"Filtering: Correct=True AND CUDA_Speedup_Native>={args.min_speedup} AND Max_Diff<={args.max_diff}...")
    kept = []
    for i, row in enumerate(ds):
        if filter_row(row, args.min_speedup, args.max_diff):
            kept.append(row)
    print(f"  {len(kept)} rows kept (out of {len(ds)})")

    if len(kept) == 0:
        print("FATAL: nothing passed the filter.", file=sys.stderr)
        sys.exit(2)

    # Deterministic shuffle + cap
    import random
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    kept = kept[: args.max_samples]
    print(f"Capped to {len(kept)} rows.")

    # Write JSONL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with out_path.open("w") as fh:
        for row in kept:
            try:
                example = build_messages(row)
            except Exception as exc:
                print(f"  skipped row (build error): {exc}")
                n_skipped += 1
                continue
            # Sanity: ensure rename happened where applicable
            assistant = example["messages"][-1]["content"]
            if "kernel_function" in assistant and "run_kernel" not in assistant:
                # rename didn't catch it — skip rather than emit broken contract
                n_skipped += 1
                continue
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} examples to {out_path} (skipped {n_skipped}).")

    # Quick provenance summary
    op_counts: dict[str, int] = {}
    for row in kept[:n_written]:
        op = str(row.get("Op_Name") or "?")
        op_counts[op] = op_counts.get(op, 0) + 1
    top = sorted(op_counts.items(), key=lambda kv: -kv[1])[:15]
    print("Top ops by frequency:")
    for op, count in top:
        print(f"  {count:4d}  {op}")


if __name__ == "__main__":
    main()
