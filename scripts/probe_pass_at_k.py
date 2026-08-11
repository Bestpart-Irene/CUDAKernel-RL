#!/usr/bin/env python3
"""EXP-018c-p0 — base-model cold-start feasibility probe (NO training).

Planner spec (research/audits/2026-08-10-planner-plan-coherence.md §6):
measure whether base Qwen3-Coder-30B-A3B, prompted with the MIGRATED
(fc6c8f1) extern "C" prompts from datasets/combined_kernelforge.jsonl,
emits parseable + nvcc-compilable extern "C" kernels at p >= 0.05 — the
exact revisit condition of do-not-repeat.md 2026-05-16 (EXP-001).

Decision rule:
  pooled p(parseable+compilable) >= 0.05  ->  "COLD-START LICENSED (p>=0.05)"
  pooled p(parseable+compilable) <  0.05  ->  "EXP-018B REQUIRED (p<0.05)"

Tasks (EXP-018a 3-task spike pool, located in the migrated dataset):
  - vector_add_e2      (task_id == "vector_add_e2", E2 signature)
  - F.elu              (single-op ops == ["F.elu"] ops_local_fallback row, E1)
  - F.softplus         (single-op ops == ["F.softplus"] row, E1)
If a named task is absent from the dataset, the closest elementary
elementwise ops6k row is substituted and LOUDLY flagged in stdout and in
summary.json ("task_substitutions").

Per task: N samples (default 32) at temperature 1.0, max_new_tokens 2048,
same sampling params as Stage 1 GRPO (top_p 0.95, top_k 50, rep 1.05).
Every completion is classified through the FROZEN extern "C" evaluator
(eval_service.eval_core.evaluate_ops6k_kernel_impl — read-only import)
into buckets:
    no_code_block   extract_cuda_code() found nothing
    parse_fail      code extracted but rejected at source scan
                    (forbidden torch::/ATen/pybind patterns)
    compile_fail    raw nvcc (no libtorch) rejected the source
    symbol_missing  compiled but `nm -D` shows no run_kernel symbol
    compile_ok      compiled + symbol present, but wrong / anti-hack /
                    reference-eval failure (detail in the record)
    correct         matched the reference on 5 seeds + anti-hack pass

Every completion + bucket is appended INCREMENTALLY (fsync'd) to
  <outdir>/completions.jsonl
so SIGTERM never loses finished samples; re-running resumes from what is
already on disk. Summary (per-task + pooled Wilson 95% CIs, pass@k,
verdict line) goes to <outdir>/summary.json and stdout.

Usage (single H200, ~1-1.5h):
    sbatch scripts/cluster/exp018c_p0.slurm
or directly:
    python scripts/probe_pass_at_k.py
Env knobs: PROBE_018C_P0_SAMPLES (32), PROBE_018C_P0_MAX_NEW_TOKENS (2048),
PROBE_018C_P0_TEMPERATURE (1.0), PROBE_018C_P0_GEN_BATCH (8),
PROBE_018C_P0_OUTDIR (outputs/probe_018c_p0), PROBE_018C_P0_SKIP_EVAL (0 —
set 1 to classify only up to extraction/source-scan on a CUDA-less box).
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same posture as scripts/debug_eval_pipeline.py: no Unsloth, local eval.
os.environ.setdefault("KERNELFORGE_SKIP_UNSLOTH", "1")
os.environ.setdefault("KERNELFORGE_EVAL_BACKEND", "local")

DATASET_PATH = ROOT / "datasets" / "combined_kernelforge.jsonl"

SAMPLES_PER_TASK = int(os.getenv("PROBE_018C_P0_SAMPLES", "32"))
MAX_NEW_TOKENS = int(os.getenv("PROBE_018C_P0_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("PROBE_018C_P0_TEMPERATURE", "1.0"))
GEN_BATCH = int(os.getenv("PROBE_018C_P0_GEN_BATCH", "8"))
OUT_DIR = Path(os.getenv("PROBE_018C_P0_OUTDIR", "outputs/probe_018c_p0"))
SKIP_EVAL = os.getenv("PROBE_018C_P0_SKIP_EVAL", "0") == "1"
SEED_BASE = int(os.getenv("PROBE_018C_P0_SEED_BASE", "20260810"))

# do-not-repeat 2026-05-16 revisit threshold.
P_THRESHOLD = 0.05

# Buckets whose members are "parseable + compilable" for the decision metric.
PARSE_COMPILE_BUCKETS = {"compile_ok", "symbol_missing", "correct"}

_STOP = {"flag": False, "signal": None}


def _handle_term(signum, _frame):
    """Finish the in-flight sample, then stop; completed work is on disk."""
    _STOP["flag"] = True
    _STOP["signal"] = signum
    print(f"\n[probe] caught signal {signum} — stopping after in-flight sample; "
          "completions.jsonl already holds every finished sample.", flush=True)


# ----------------------------------------------------------------------
# Task selection
# ----------------------------------------------------------------------

def _load_dataset_rows() -> list[dict]:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _single_op_row(rows: list[dict], op: str) -> dict | None:
    """Exact single-op ops6k row match (e.g. ops == ["F.elu"])."""
    from training.task_support import normalize_task_row

    for row in rows:
        norm = normalize_task_row(row)
        if norm["evaluation_backend"] == "ops6k" and norm["ops"] == [op]:
            return row
    return None


def select_probe_tasks(rows: list[dict]) -> tuple[list[tuple[str, dict]], list[str]]:
    """Return [(task_key, raw_row)] for the 3-task spike pool + substitution flags."""
    from training.task_support import normalize_task_row

    substitutions: list[str] = []
    picked: list[tuple[str, dict]] = []
    taken_ids: set[int] = set()

    def _elementwise_fallback(wanted: str) -> dict | None:
        # Closest elementary elementwise ops among live ops6k rows, easiest first.
        candidates = ["F.elu", "F.softplus", "torch.add", "F.relu", "torch.abs",
                      "torch.tanh", "F.hardswish"]
        for op in candidates:
            row = _single_op_row(rows, op)
            if row is not None and id(row) not in taken_ids:
                substitutions.append(
                    f"TASK SUBSTITUTION: {wanted!r} not found in "
                    f"{DATASET_PATH.name}; using closest elementary "
                    f"elementwise row ops={op!r} instead."
                )
                return row
        return None

    # 1. vector_add_e2 by explicit task_id.
    va = next((r for r in rows if str(r.get("task_id") or "") == "vector_add_e2"), None)
    if va is None:
        va = _elementwise_fallback("vector_add_e2")
    if va is not None:
        picked.append(("vector_add_e2", va))
        taken_ids.add(id(va))

    # 2/3. F.elu and F.softplus single-op rows.
    for key, op in (("f_elu", "F.elu"), ("f_softplus", "F.softplus")):
        row = _single_op_row(rows, op)
        if row is not None and id(row) in taken_ids:
            row = None
        if row is None:
            row = _elementwise_fallback(op)
        if row is not None:
            picked.append((key, row))
            taken_ids.add(id(row))

    if len(picked) < 3:
        substitutions.append(
            f"FATAL-ADJACENT: only {len(picked)}/3 probe tasks could be "
            "resolved from the dataset — probe statistics will be underpowered."
        )

    # Preflight: every picked task must have an inferable extern-C signature,
    # otherwise the evaluator rejects 100% of samples for a task-side reason.
    for key, row in picked:
        norm = normalize_task_row(row)
        from training.task_support import infer_signature_class
        sig = (norm.get("extern_c_signature") or {}).get("class") or \
            infer_signature_class(norm.get("task_code") or "")
        if sig not in {"E1", "E2"}:
            substitutions.append(
                f"WARNING: task {key!r} has uninferable extern-C signature "
                f"({sig!r}) — every completion would fail at the evaluator "
                "for a task-side reason, not a policy-side one."
            )
    return picked, substitutions


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

def classify_completion(completion_text: str, task_row: dict) -> dict:
    """Extract + run one completion through the frozen extern-C evaluator.

    Returns dict with keys: bucket, extracted_code, eval (subset) — computed
    as far as the harness allows on this host.
    """
    from eval_service.eval_core import evaluate_ops6k_kernel_impl
    from openenv_env.anti_hack import scan_source_forbidden
    from training.multi_turn_rollout import extract_cuda_code
    from training.task_support import normalize_task_row

    record: dict = {"bucket": None, "extracted_code": None, "eval": None}

    code = extract_cuda_code(completion_text)
    if not code:
        record["bucket"] = "no_code_block"
        return record
    record["extracted_code"] = code

    # Source-scan first so a CUDA-less dry run can still separate parse_fail.
    reject = scan_source_forbidden(code)
    if reject:
        record["bucket"] = "parse_fail"
        record["eval"] = {"error": f"Source rejected: {reject}"[:500]}
        return record

    if SKIP_EVAL:
        record["bucket"] = "not_evaluated"
        return record

    norm = normalize_task_row(task_row)
    result = evaluate_ops6k_kernel_impl({
        "cuda_code": code,
        "task_code": norm.get("task_code") or "",
        "warmup_iters": 3,
        "benchmark_runs": 3,
        "evaluation_backend": "ops6k",
        "extern_c_signature": norm.get("extern_c_signature"),
    })

    err = str(result.get("error") or "")
    vmsg = str(result.get("verifier_msg") or "")
    record["eval"] = {
        "compiles": bool(result.get("compiles")),
        "correct": bool(result.get("correct")),
        "verifier_msg": vmsg[:500],
        "error": err[:800],
    }

    if err == "nvcc not in PATH":
        # Infra failure — attributing it to the policy would poison the probe.
        raise RuntimeError(
            "nvcc not in PATH inside the evaluator — the CUDA module is not "
            "loaded. Fix the environment (cluster_paths.sh loads cuda/12.8.0) "
            "and re-run; probe aborted so infra failure is not recorded as "
            "compile_fail."
        )

    if err.startswith("Source rejected:"):
        record["bucket"] = "parse_fail"
    elif not result.get("compiles"):
        record["bucket"] = "compile_fail"
    elif "undefined symbol: run_kernel" in (vmsg + err):
        record["bucket"] = "symbol_missing"
    elif result.get("correct"):
        record["bucket"] = "correct"
    else:
        record["bucket"] = "compile_ok"
    return record


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval (pure math — no scipy dependency)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(records: list[dict], task_keys: list[str],
              substitutions: list[str], partial: bool) -> dict:
    from evaluation.pass_at_k import pass_at_k

    def _stats(recs: list[dict]) -> dict:
        n = len(recs)
        buckets: dict[str, int] = {}
        for r in recs:
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
        pc = sum(1 for r in recs if r["bucket"] in PARSE_COMPILE_BUCKETS)
        sym = sum(1 for r in recs
                  if r["bucket"] in {"compile_ok", "correct"})
        cor = sum(1 for r in recs if r["bucket"] == "correct")
        lo, hi = wilson_ci(pc, n)
        out = {
            "n": n,
            "buckets": buckets,
            "parse_compile": pc,
            "p_parse_compile": (pc / n) if n else 0.0,
            "p_parse_compile_wilson95": [lo, hi],
            "symbol_present": sym,
            "correct": cor,
            "p_correct": (cor / n) if n else 0.0,
            "p_correct_wilson95": list(wilson_ci(cor, n)),
        }
        if n:
            out["pass_at_k_correct"] = {
                f"pass@{k}": pass_at_k(n=n, c=cor, k=k)
                for k in (1, 8, 32) if k <= n
            }
        return out

    per_task = {}
    for key in task_keys:
        per_task[key] = _stats([r for r in records if r["task_key"] == key])
    pooled = _stats(records)

    p = pooled["p_parse_compile"]
    verdict = (f"COLD-START LICENSED (p>={P_THRESHOLD})" if p >= P_THRESHOLD
               else f"EXP-018B REQUIRED (p<{P_THRESHOLD})")
    lo, hi = pooled["p_parse_compile_wilson95"]
    ci_note = ""
    if lo < P_THRESHOLD <= hi:
        ci_note = (f"NOTE: Wilson 95% CI [{lo:.3f}, {hi:.3f}] straddles "
                   f"{P_THRESHOLD} — verdict is on the point estimate and "
                   "is not statistically resolved at this sample size.")

    return {
        "experiment": "EXP-018c-p0",
        "partial": partial,
        "interrupted_by_signal": _STOP["signal"],
        "samples_per_task_target": SAMPLES_PER_TASK,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "threshold": P_THRESHOLD,
        "task_substitutions": substitutions,
        "per_task": per_task,
        "pooled": pooled,
        "verdict": verdict,
        "ci_note": ci_note,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _append_jsonl(path: Path, record: dict) -> None:
    """Append one record and fsync so SIGKILL cannot lose it."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_done(path: Path) -> dict[tuple[str, int], dict]:
    done: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(rec["task_key"], int(rec["sample_idx"]))] = rec
            except Exception:
                continue  # half-written trailing line from a previous kill
    return done


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completions_path = OUT_DIR / "completions.jsonl"
    summary_path = OUT_DIR / "summary.json"

    print("=" * 72)
    print("EXP-018c-p0 — base-model extern-C cold-start feasibility probe")
    print("=" * 72)
    print(f"dataset          : {DATASET_PATH}")
    print(f"samples/task     : {SAMPLES_PER_TASK}")
    print(f"temperature      : {TEMPERATURE}")
    print(f"max_new_tokens   : {MAX_NEW_TOKENS}")
    print(f"gen batch        : {GEN_BATCH}")
    print(f"output           : {completions_path}")
    print(f"skip_eval        : {SKIP_EVAL}")

    rows = _load_dataset_rows()
    tasks, substitutions = select_probe_tasks(rows)
    for flag in substitutions:
        print(f"!!! {flag}")
    task_keys = [k for k, _ in tasks]
    print(f"tasks            : {task_keys}")
    if not tasks:
        print("FATAL: no probe tasks resolved.")
        return 2

    done = _load_done(completions_path)
    if done:
        print(f"resume           : {len(done)} finished samples found on disk; skipping those.")

    print("\n[setup] loading BASE model (no SFT checkpoint — cold-start probe)...")
    import torch
    from training.model_loader import PRIMARY_MODEL, load_model_and_tokenizer
    from training.task_support import build_generation_prompt

    print(f"[setup] model = {PRIMARY_MODEL} (KERNELFORGE_MODEL to override)")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=None)
    model.eval()

    records: list[dict] = list(done.values())

    for task_idx, (task_key, raw_row) in enumerate(tasks):
        prompt_text = build_generation_prompt(raw_row)
        messages = [
            {"role": "system",
             "content": "You are a CUDA kernel expert. Write a complete CUDA "
                        "kernel for the requested operator."},
            {"role": "user", "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        print(f"\n=== task {task_key}: prompt {prompt_len} tokens ===")

        pending = [i for i in range(SAMPLES_PER_TASK)
                   if (task_key, i) not in done]
        chunk_no = 0
        while pending and not _STOP["flag"]:
            batch = pending[:GEN_BATCH]
            pending = pending[GEN_BATCH:]
            # Deterministic across runs (str hash() is salted per process).
            seed = SEED_BASE + 10_000 * task_idx + 100 * chunk_no
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.05,
                    num_return_sequences=len(batch),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_s = time.time() - t0
            for j, sample_idx in enumerate(batch):
                completion_ids = gen[j][prompt_len:]
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                cls = classify_completion(text, raw_row)
                rec = {
                    "experiment": "EXP-018c-p0",
                    "task_key": task_key,
                    "task_id": raw_row.get("task_id"),
                    "ops": raw_row.get("ops"),
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "gen_seconds_batch": round(gen_s, 2),
                    "n_completion_tokens": int(completion_ids.shape[0]),
                    "bucket": cls["bucket"],
                    "eval": cls["eval"],
                    "extracted_code": cls["extracted_code"],
                    "completion": text,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _append_jsonl(completions_path, rec)
                records.append(rec)
                print(f"  [{task_key} {sample_idx + 1}/{SAMPLES_PER_TASK}] "
                      f"bucket={cls['bucket']} "
                      f"({int(completion_ids.shape[0])} tok, batch {gen_s:.0f}s)",
                      flush=True)
                if _STOP["flag"]:
                    break
            chunk_no += 1
        if _STOP["flag"]:
            break

    partial = _STOP["flag"] or any(
        sum(1 for r in records if r["task_key"] == k) < SAMPLES_PER_TASK
        for k in task_keys
    )
    summary = summarize(records, task_keys, substitutions, partial)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("EXP-018c-p0 SUMMARY" + ("  (PARTIAL RUN)" if partial else ""))
    print("=" * 72)
    for key in task_keys:
        st = summary["per_task"][key]
        lo, hi = st["p_parse_compile_wilson95"]
        print(f"  {key:16s} n={st['n']:3d} buckets={st['buckets']} "
              f"p(parse+compile)={st['p_parse_compile']:.3f} "
              f"[{lo:.3f}, {hi:.3f}] correct={st['correct']}")
    pooled = summary["pooled"]
    lo, hi = pooled["p_parse_compile_wilson95"]
    print(f"  {'POOLED':16s} n={pooled['n']:3d} "
          f"p(parse+compile)={pooled['p_parse_compile']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] correct={pooled['correct']}")
    if summary["ci_note"]:
        print(f"  {summary['ci_note']}")
    print(f"\nVERDICT: {summary['verdict']}")
    print(f"summary  -> {summary_path}")
    print(f"raw data -> {completions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
