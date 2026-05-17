"""Diagnose where the EXP-003 verification's reward=-1 signal is coming from.

EXP-004 update: iterate over ALL WCC-filtered stage1 tasks (4 rows when
KERNELFORGE_STAGE1_BACKEND_FILTER=wcc), running one rollout per task, and
dump per-task structured JSON so we can identify which bucket inside
compute_reward's -1 branch is firing:
  - compile_failed     (nvcc returns non-zero on extracted code)
  - compiled_but_wrong (compiles=True, correct=False, anti-hack not tripped)
  - anti_hack          (eval error message starts with "Anti-hack:")
  - eval_crash         (evaluate_code_remote raised, or eval result["error"]
                         is a non-anti-hack, non-compile message)

Loads the EXP-002 SFT adapter (outputs/kernelforge-stage2/checkpoint-50/),
mirrors stage1_warmup.py's KERNELFORGE_STAGE1_BACKEND_FILTER behavior, and
walks the full reward chain step-by-step for every filtered task.

Run as a slurm job: `sbatch scripts/cluster/debug_eval.slurm`.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KERNELFORGE_SKIP_UNSLOTH", "1")
os.environ.setdefault("KERNELFORGE_EVAL_BACKEND", "local")


def _classify_bucket(
    extracted_ok: bool,
    local_compile_ok: bool,
    eval_compiles: bool | None,
    eval_correct: bool | None,
    eval_error: str,
    eval_crashed: bool,
) -> str:
    """Categorize a rollout into one of the 4 EXP-004 diagnostic buckets."""
    if eval_crashed:
        return "eval_crash"
    if not extracted_ok:
        # Extraction failure presents as compile failure downstream.
        return "compile_failed"
    if local_compile_ok is False:
        return "compile_failed"
    if eval_compiles is False:
        return "compile_failed"
    err_str = (eval_error or "").strip()
    if err_str.startswith("Anti-hack:"):
        return "anti_hack"
    if eval_correct is False:
        return "compiled_but_wrong"
    if eval_correct is True:
        # Not a -1 bucket; reward chain succeeded.
        return "ok_correct"
    # Compiled but unknown correctness + non-anti-hack error → treat as crash.
    return "eval_crash"


def _safe(val):
    """JSON-safe coerce — keep simple types, stringify the rest."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _safe(v) for k, v in val.items()}
    return str(val)


def main() -> None:
    print("=" * 72)
    print("EXP-004 WCC failure-bucket diagnostic — 1 rollout per WCC task")
    print("=" * 72)

    backend_filter = (os.getenv("KERNELFORGE_STAGE1_BACKEND_FILTER") or "").strip()
    print(f"KERNELFORGE_STAGE1_BACKEND_FILTER = {backend_filter!r}")
    print(f"KERNELFORGE_EVAL_BACKEND          = {os.environ.get('KERNELFORGE_EVAL_BACKEND')!r}")

    out_dir = Path(os.getenv("KERNELFORGE_EXP004_OUTDIR", "outputs/exp004_debug"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Per-task JSON dump directory      = {out_dir}")

    print("\n[setup 1/3] Loading model + tokenizer with EXP-002 SFT adapter...")
    from training.model_loader import load_model_and_tokenizer
    ckpt = "outputs/kernelforge-stage2/checkpoint-50"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=ckpt)
    print(f"  loaded; tokenizer.eos_token={tokenizer.eos_token!r}")

    print("\n[setup 2/3] Loading filtered stage1 prompts...")
    from training.stage1_warmup import load_stage1_dataset
    from training.task_support import normalize_task_row
    ds = load_stage1_dataset()
    raw_rows = ds.to_list() if hasattr(ds, "to_list") else list(ds)

    # Re-apply filter defensively in case load_stage1_dataset fell back to a
    # different dataset (e.g., the unfiltered fallback when filter yields 0).
    if backend_filter:
        filtered_rows = []
        for r in raw_rows:
            backend = str(r.get("evaluation_backend", "")).strip()
            if not backend:
                # Infer via normalization if not already set on the row.
                backend = normalize_task_row(r).get("evaluation_backend", "")
            if backend == backend_filter:
                filtered_rows.append(r)
        print(f"  rows loaded={len(raw_rows)}; after filter={len(filtered_rows)}")
    else:
        filtered_rows = raw_rows
        print(f"  rows loaded={len(raw_rows)} (no filter)")

    if not filtered_rows:
        print("  ❌ no rows after filter — aborting.")
        return

    print("\n[setup 3/3] Imports for rollout / extract / compile / eval...")
    import torch
    from training.multi_turn_rollout import (
        extract_cuda_code,
        _local_compile_check,
        evaluate_code_remote,
    )

    bucket_counts = {
        "compile_failed": 0,
        "compiled_but_wrong": 0,
        "anti_hack": 0,
        "eval_crash": 0,
        "ok_correct": 0,
    }

    for task_idx, raw_row in enumerate(filtered_rows):
        task = normalize_task_row(raw_row)
        op_name = (
            (task.get("ops") or ["unknown"])[0]
            if isinstance(task.get("ops"), list) and task.get("ops")
            else str(task.get("ops") or task.get("data_source") or "unknown")
        )
        prompt_text = task.get("prompt") or task.get("task_text") or str(task)

        print("\n" + "=" * 72)
        print(f"[task {task_idx + 1}/{len(filtered_rows)}] op={op_name!r}  "
              f"data_source={task.get('data_source')!r}  backend={task.get('evaluation_backend')!r}")
        print("=" * 72)
        print(f"  prompt[0..300]: {prompt_text[:300]}...")
        print(f"  supports_evaluation: {task.get('supports_evaluation')}")

        record: dict = {
            "task_index": task_idx,
            "op_name": op_name,
            "data_source": task.get("data_source"),
            "evaluation_backend": task.get("evaluation_backend"),
            "prompt": prompt_text[:500],
            "completion_full": None,
            "extracted_code": None,
            "local_compile_ok": None,
            "local_compile_err": None,
            "eval_compiles": None,
            "eval_correct": None,
            "eval_error": None,
            "eval_reward_in_result": None,
            "eval_speedup_vs_orig": None,
            "eval_speedup_vs_dg": None,
            "anti_hack_verdicts": None,
            "raw_eval_result_keys": None,
            "bucket": None,
            "exception": None,
        }

        # --- Generate one completion ---
        print("\n  [step 1/4] Generating ONE completion (max_new_tokens=1024)...")
        messages = [
            {"role": "system", "content": "You are a CUDA kernel expert. Write a complete CUDA kernel for the requested operator."},
            {"role": "user", "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = gen[0][inputs["input_ids"].shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        record["completion_full"] = completion_text
        eos_seen = tokenizer.eos_token_id in completion_ids.tolist()
        print(f"    completion: {len(completion_text)} chars, {len(completion_ids)} tokens, EOS={eos_seen}")
        print(f"    --- first 300 chars ---\n{completion_text[:300]}")
        print(f"    --- last 300 chars ---\n{completion_text[-300:]}")

        # --- Extract CUDA code ---
        print("\n  [step 2/4] extract_cuda_code()...")
        code = extract_cuda_code(completion_text)
        if not code:
            print("    ❌ NO CUDA CODE EXTRACTED")
            record["extracted_code"] = None
            record["local_compile_ok"] = False
            record["local_compile_err"] = "extract_cuda_code returned empty"
            record["bucket"] = _classify_bucket(False, False, None, None, "", False)
            bucket_counts[record["bucket"]] += 1
            _write_task_json(out_dir, task_idx, op_name, record)
            continue
        record["extracted_code"] = code
        print(f"    ✅ extracted {len(code)} chars")

        # --- Local compile check ---
        print("\n  [step 3/4] _local_compile_check()...")
        compiles_ok, compile_err = _local_compile_check(code)
        record["local_compile_ok"] = bool(compiles_ok)
        record["local_compile_err"] = (compile_err or "")[:2000]
        if not compiles_ok:
            print(f"    ❌ local compile FAILED (first 400 chars): {compile_err[:400]}")
            record["bucket"] = _classify_bucket(True, False, None, None, "", False)
            bucket_counts[record["bucket"]] += 1
            _write_task_json(out_dir, task_idx, op_name, record)
            continue
        print(f"    ✅ compiled (nvcc -arch={os.getenv('KERNELFORGE_TARGET_ARCH', 'sm_80')})")

        # --- Remote eval ---
        print("\n  [step 4/4] evaluate_code_remote() (backend=local)...")
        eval_crashed = False
        try:
            result = evaluate_code_remote(
                code,
                task,
                baseline_orig_ms=None,
                baseline_dg_ms=None,
            )
            record["raw_eval_result_keys"] = sorted([str(k) for k in result.keys()])
            record["eval_compiles"] = bool(result.get("compiles", False))
            record["eval_correct"] = bool(result.get("correct", False))
            err_text = result.get("error") or ""
            record["eval_error"] = str(err_text)  # untruncated, per spec
            record["eval_reward_in_result"] = _safe(result.get("reward"))
            record["eval_speedup_vs_orig"] = _safe(
                result.get("speedup_vs_orig") or result.get("speedup_vs_eager")
            )
            record["eval_speedup_vs_dg"] = _safe(
                result.get("speedup_vs_dg") or result.get("speedup_vs_compile")
            )
            # Surface anti-hack info if the eval embedded any (best-effort).
            ah_keys = [k for k in result.keys() if "anti" in str(k).lower() or "hack" in str(k).lower()]
            if ah_keys:
                record["anti_hack_verdicts"] = {k: _safe(result.get(k)) for k in ah_keys}
            elif str(err_text).startswith("Anti-hack:"):
                record["anti_hack_verdicts"] = {"error_message": str(err_text)}

            print(f"    keys: {record['raw_eval_result_keys']}")
            print(f"    compiles={record['eval_compiles']} correct={record['eval_correct']}")
            print(f"    speedup_vs_orig={record['eval_speedup_vs_orig']} "
                  f"speedup_vs_dg={record['eval_speedup_vs_dg']}")
            print(f"    reward_in_result={record['eval_reward_in_result']}")
            if err_text:
                print(f"    error (untruncated):")
                for line in str(err_text).splitlines():
                    print(f"      {line}")
            else:
                print("    error: (none)")
        except Exception as exc:
            eval_crashed = True
            record["exception"] = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            record["eval_error"] = tb  # full untruncated traceback
            print(f"    ❌ evaluate_code_remote raised: {record['exception']}")
            print(tb)

        record["bucket"] = _classify_bucket(
            extracted_ok=True,
            local_compile_ok=True,
            eval_compiles=record["eval_compiles"],
            eval_correct=record["eval_correct"],
            eval_error=record["eval_error"] or "",
            eval_crashed=eval_crashed,
        )
        bucket_counts[record["bucket"]] += 1
        print(f"\n  → bucket: {record['bucket']}")
        _write_task_json(out_dir, task_idx, op_name, record)

    print("\n" + "=" * 72)
    print("EXP-004 SUMMARY")
    print("=" * 72)
    print(
        "Bucket counts: "
        f"compile_failed={bucket_counts['compile_failed']}, "
        f"compiled_but_wrong={bucket_counts['compiled_but_wrong']}, "
        f"anti_hack={bucket_counts['anti_hack']}, "
        f"eval_crash={bucket_counts['eval_crash']}"
    )
    if bucket_counts.get("ok_correct", 0):
        print(f"(also ok_correct={bucket_counts['ok_correct']} — surprising; not a -1 case)")
    print(f"Per-task dumps in: {out_dir}")
    print("=" * 72)


def _write_task_json(out_dir: Path, task_idx: int, op_name: str, record: dict) -> None:
    """Write per-task JSON dump for offline inspection."""
    safe_op = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(op_name))[:80]
    fname = out_dir / f"{task_idx:02d}_{safe_op}.json"
    try:
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str, ensure_ascii=False)
        print(f"  → wrote {fname}")
    except Exception as exc:
        print(f"  ⚠ failed to write {fname}: {exc}")


if __name__ == "__main__":
    main()
