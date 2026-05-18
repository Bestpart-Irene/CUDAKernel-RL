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


_PYBIND_DEF_RE = re.compile(
    r'm\.def\s*\(\s*"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"\s*,\s*&(?P=name)\b'
)


def rename_sakana_entrypoint_to_run_kernel(code: str) -> str:
    """Rewrite Sakana's pybind entry point → `run_kernel`.

    Strategy: find the actual entrypoint by parsing the
    `m.def("NAME", &NAME, ...)` line. Only rename that exact symbol
    whole-word. Avoids touching unrelated identifiers in comments or
    helper functions. Always also normalizes the older `kernel_function`
    name for backwards compatibility with earlier Sakana exports.
    """
    code = re.sub(r"\bkernel_function\b", "run_kernel", code)
    match = _PYBIND_DEF_RE.search(code)
    if not match:
        # No pybind def found — fall back to whole-word forward (legacy).
        return re.sub(r"\bforward\b", "run_kernel", code)
    name = match.group("name")
    if name == "run_kernel":
        return code
    return re.sub(rf"\b{re.escape(name)}\b", "run_kernel", code)


def build_messages(row: dict, target_gpu: str = "A100", target_arch: str = "sm_80") -> dict:
    """Compose one HF-messages SFT example from a Sakana row."""
    pytorch_ref = str(row.get("PyTorch_Code_Module") or "").strip()
    cuda_code = rename_sakana_entrypoint_to_run_kernel(str(row.get("CUDA_Code") or "").strip())
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


def filter_row(
    row: dict,
    min_speedup: float,
    max_diff: float,
    max_code_tokens: int,
    tokenizer,
) -> bool:
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
    # Real token-length filter (was char-based, which silently undershoots
    # for CUDA code where token/char ratio runs higher than the rule-of-
    # thumb 3.5). Without this, EXP-002b saw assistant messages clipped at
    # max_completion_length=1024 even after the char-cap.
    n_tokens = len(tokenizer.encode(cuda_code, add_special_tokens=False))
    if n_tokens > max_code_tokens:
        return False
    if "forward" in cuda_code or "kernel_function" in cuda_code or "run_kernel" in cuda_code:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1500,
                        help="Maximum SFT rows to emit (default 1500).")
    parser.add_argument("--min-speedup", type=float, default=0.5,
                        help="Minimum CUDA_Speedup_Native to keep a row.")
    parser.add_argument("--max-diff", type=float, default=1e-2,
                        help="Maximum Max_Diff to keep a row (correctness margin).")
    parser.add_argument("--max-code-tokens", type=int, default=900,
                        help="Maximum CUDA_Code length in tokens (training "
                             "tokenizer). Stage 1 max_completion_length=1024; "
                             "leave headroom for the assistant fence + prompt.")
    parser.add_argument("--tokenizer", type=str,
                        default="unsloth/Qwen3-Coder-30B-A3B-Instruct",
                        help="HF model id whose tokenizer is used for the "
                             "length filter. MUST match the training tokenizer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=str, default="level_1,level_2",
                        help="Comma-separated split names (e.g. 'level_1' or 'level_1,level_2').")
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--underfill-warn-ratio", type=float, default=0.5,
                        help="If final row count < ratio * max_samples, "
                             "WARN loudly. 0 disables the check.")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {args.tokenizer} for length filter...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    hf_datasets = _load_hf_datasets()
    print(f"Loading SakanaAI/AI-CUDA-Engineer-Archive splits={args.levels}...")
    splits = [s.strip() for s in args.levels.split(",") if s.strip()]
    parts = []
    for split in splits:
        d = hf_datasets.load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=split)
        print(f"  {split}: {len(d)} rows")
        parts.append(d)
    ds = hf_datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    print(f"Filtering: Correct=True AND CUDA_Speedup_Native>={args.min_speedup} "
          f"AND Max_Diff<={args.max_diff} AND tokens(CUDA_Code)<={args.max_code_tokens}...")
    kept = []
    for i, row in enumerate(ds):
        if filter_row(row, args.min_speedup, args.max_diff,
                      args.max_code_tokens, tokenizer):
            kept.append(row)
    print(f"  {len(kept)} rows kept (out of {len(ds)})")

    if len(kept) == 0:
        print("FATAL: nothing passed the filter.", file=sys.stderr)
        sys.exit(2)

    # Deduplicate on the renamed CUDA source. Sakana L1 + L2 frequently
    # contain near-identical kernel variants for the same op; without
    # this the SFT corpus is biased toward dense ops.
    seen: set[str] = set()
    deduped = []
    for row in kept:
        renamed = rename_sakana_entrypoint_to_run_kernel(
            str(row.get("CUDA_Code") or "")
        )
        key = renamed.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    n_dropped_dup = len(kept) - len(deduped)
    if n_dropped_dup:
        print(f"  deduped: dropped {n_dropped_dup} near-duplicate kernels "
              f"(kept {len(deduped)})")
    kept = deduped

    # Deterministic shuffle + cap
    import random
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    kept = kept[: args.max_samples]
    print(f"Capped to {len(kept)} rows.")

    if args.underfill_warn_ratio > 0 and len(kept) < args.max_samples * args.underfill_warn_ratio:
        print(
            f"WARNING: under-fill — only {len(kept)} rows survived filter+dedup "
            f"vs --max-samples={args.max_samples}. Loosen filters (lower "
            f"--min-speedup, raise --max-diff or --max-code-tokens) or accept "
            f"a smaller corpus.",
            file=sys.stderr,
        )

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
