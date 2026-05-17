"""Diagnose where the EXP-003 verification's reward=-1 signal is coming from.

Loads the EXP-002 SFT adapter (outputs/kernelforge-stage2/checkpoint-50/),
takes ONE stage1 prompt, generates ONE completion, then walks the full
reward chain step-by-step printing what each layer sees.

This isolates two hypotheses:
  A. SFT format adequate but reward path broken (local eval / compile fails)
  B. SFT format inadequate — model emits non-extractable / wrong-arch code

Run as a slurm job: `sbatch scripts/cluster/debug_eval.slurm`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KERNELFORGE_SKIP_UNSLOTH", "1")
os.environ.setdefault("KERNELFORGE_EVAL_BACKEND", "local")


def main() -> None:
    print("=" * 72)
    print("EXP-003 verification debug — single-rollout reward chain probe")
    print("=" * 72)

    print("\n[1/6] Loading model + tokenizer with EXP-002 SFT adapter...")
    from training.model_loader import load_model_and_tokenizer
    ckpt = "outputs/kernelforge-stage2/checkpoint-50"
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=ckpt)
    print(f"  loaded; tokenizer.eos_token={tokenizer.eos_token!r}")

    print("\n[2/6] Loading one stage1 prompt...")
    from training.stage1_warmup import load_stage1_dataset
    from training.task_support import normalize_task_row
    ds = load_stage1_dataset()
    rows = ds.to_list() if hasattr(ds, "to_list") else list(ds)
    task = normalize_task_row(rows[0])
    prompt_text = task.get("prompt") or task.get("task_text") or str(task)
    print(f"  prompt[0]: {prompt_text[:300]}...")
    print(f"  ops: {task.get('ops')}, data_source: {task.get('data_source')}")
    print(f"  supports_evaluation: {task.get('supports_evaluation')}")

    print("\n[3/6] Building chat-templated prompt + generating ONE completion (max 1024 tokens)...")
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
    import torch
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
    completion_ids = gen[0][inputs["input_ids"].shape[1] :]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    print(f"  completion length: {len(completion_text)} chars, {len(completion_ids)} tokens")
    eos_seen = tokenizer.eos_token_id in completion_ids.tolist()
    print(f"  natural EOS seen: {eos_seen}")
    print(f"  --- completion first 500 chars ---")
    print(completion_text[:500])
    print(f"  --- completion last 500 chars ---")
    print(completion_text[-500:])

    print("\n[4/6] extract_cuda_code() on completion...")
    from training.multi_turn_rollout import extract_cuda_code, _local_compile_check
    code = extract_cuda_code(completion_text)
    if not code:
        print("  ❌ NO CUDA CODE EXTRACTED.")
        print("  This means hypothesis B: SFT format inadequate / regex mismatch.")
        print("  Looking for ```cuda...``` or ```cpp...``` blocks; printing first 200 chars of completion:")
        print(f"  {completion_text[:200]!r}")
        return
    print(f"  ✅ extracted {len(code)} chars of CUDA code")
    print(f"  --- code first 400 chars ---")
    print(code[:400])

    print("\n[5/6] _local_compile_check() on extracted code (nvcc -arch=sm_80)...")
    compiles_ok, compile_err = _local_compile_check(code)
    if not compiles_ok:
        print(f"  ❌ COMPILE FAILED on H200 with -arch=sm_80")
        print(f"  Error (first 400 chars): {compile_err[:400]}")
        print("  This is hypothesis A: SFT format works but code is wrong / mis-targeted.")
        return
    print(f"  ✅ compiled successfully via nvcc -arch=sm_80")

    print("\n[6/6] evaluate_code_remote() through eval_backend=local...")
    from training.multi_turn_rollout import evaluate_code_remote
    try:
        result = evaluate_code_remote(
            code,
            task,
            baseline_orig_ms=None,
            baseline_dg_ms=None,
        )
        print(f"  eval result keys: {sorted(result.keys())}")
        print(f"  compiles: {result.get('compiles')}")
        print(f"  correct: {result.get('correct')}")
        print(f"  reward_in_result: {result.get('reward')}")
        speedup_orig = result.get("speedup_vs_orig") or result.get("speedup_vs_eager")
        speedup_compile = result.get("speedup_vs_compile")
        print(f"  speedup_vs_orig: {speedup_orig}, speedup_vs_compile: {speedup_compile}")
        err = result.get("error")
        if err:
            print(f"  ❌ eval reported error: {err[:400]}")
            print("  Hypothesis B candidate: eval path runs but kernel fails at exec.")
        elif result.get("correct"):
            print("  ✅ eval reported CORRECT — reward chain works, just first roll lucky.")
        else:
            print("  ❌ eval succeeded but kernel produced wrong output.")
    except Exception as exc:
        print(f"  ❌ evaluate_code_remote raised: {exc}")
        print("  This is hypothesis B: local eval dispatch is broken on this stack.")

    print("\n" + "=" * 72)
    print("Done. Verdict above tells us which hypothesis (A or B) explains EXP-003 reward=-1.")
    print("=" * 72)


if __name__ == "__main__":
    main()
