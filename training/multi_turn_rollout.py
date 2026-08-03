"""
Multi-turn rollout for GRPOTrainer.

The policy generates on the training GPU, while correctness and runtime reward
are computed remotely on the target A100 via CoreWeave/Northflank (or Modal).

DEAD CODE WARNING (verified 2026-05-18 against pinned trl==0.29.0):
`make_multi_turn_rollout` / `rollout_func` below is NOT invoked by TRL 0.29.
Job 6888764 log shows `[ROLLOUT_FACTORY]` once and `[ROLLOUT_CALL]` zero
times. TRL emits a runtime warning at trainer init confirming the
`rollout_func` parameter is experimental. The active reward path is
`reward_from_env(...)` below — TRL calls it directly with completions, and
the `env_reward len=0` branch inline-evaluates each completion via local
compile + remote/local eval. Implications:
  - All training is effectively single-turn regardless of
    `KERNELFORGE_STAGE1_MAX_TURNS` / `KERNELFORGE_STAGE3_MAX_TURNS`.
  - Multi-turn attribution / feedback / `[ROLLOUT prompt=...]` debug is
    dead. Diagnostic prints have been relocated to `reward_from_env`
    inline branch (commit 469f6a4) — use the `[ROLLOUT_INLINE]` and
    `[VERIFIER_INLINE]` tags.
  - Re-enabling multi-turn requires either pinning to a TRL version that
    still calls rollout_func, or driving the turn loop manually inside
    `reward_from_env`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from training.curriculum import format_topology_context
from training.run_metadata import utc_timestamp_rfc3339
from training.task_support import (
    build_generation_prompt,
    build_prompt_lookup,
    compute_task_reward,
    evaluate_code_remote,
    normalize_eval_result,
    normalize_task_row,
)
LOCAL_COMPILE_CHECK = os.getenv("KERNELFORGE_LOCAL_COMPILE", "1") == "1"
TARGET_CUDA_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
MAX_FEEDBACK_CHARS = int(os.getenv("KERNELFORGE_MAX_FEEDBACK_CHARS", "1200"))
MAX_ERROR_CHARS = int(os.getenv("KERNELFORGE_MAX_ERROR_CHARS", "800"))
ROLLOUT_LOG_PATH = Path(
    os.getenv("KERNELFORGE_ROLLOUT_LOG", "outputs/rollout_metrics.jsonl")
).resolve()


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from model output (fenced block or raw __global__).

    Handles the truncated-completion case: when max_new_tokens cuts the
    generation off before the closing ``` fence, still return the body
    after the opening fence (without it nvcc sees ```cpp on line 1 and
    fails with "unrecognized token"). EXP-004 v2 caught this.
    """
    # Order matters: longer markers first so ```cpp/```cuda match before ```c.
    for marker in ["```cuda", "```cpp", "```c++", "```c"]:
        idx = text.find(marker)
        if idx == -1:
            continue
        start = idx + len(marker)
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
        # Fence opened but not closed (likely truncated). Drop the opener
        # but keep whatever body the model emitted.
        return text[start:].strip()

    # Untagged ``` fence (or a tag not in the list above): take the first
    # fenced body that looks like kernel code. Without this, the raw-regex
    # fallback below returned the WHOLE text including the backtick lines.
    fenced = re.search(r"```[^\n`]*\n(.*?)(?:\n```|$)", text, re.DOTALL)
    if fenced:
        body = fenced.group(1).strip()
        if (
            re.search(r"__global__\s+void\s+\w+", body)
            or "PYBIND11_MODULE" in body
            or 'extern "C"' in body
        ):
            return body

    if re.search(r"__global__\s+void\s+\w+", text) or "PYBIND11_MODULE" in text:
        # Strip stray fence lines so nvcc never sees backticks.
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return ""


def _append_rollout_log(record: dict[str, Any]) -> None:
    """Append one rollout event to a JSONL log file."""
    try:
        ROLLOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload.setdefault("timestamp", utc_timestamp_rfc3339())
        with ROLLOUT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def _local_compile_check(code: str) -> tuple[bool, str]:
    """Quick local nvcc syntax check before paying for a remote evaluation."""
    if not LOCAL_COMPILE_CHECK:
        return True, ""

    # EXP-014 (2026-05-25): ops6k kernels include `<torch/extension.h>` which
    # bare nvcc can't find without torch's include paths. The remote ops6k
    # eval (`eval_service/eval_core.py:evaluate_ops6k_kernel_impl`) uses
    # `torch.utils.cpp_extension.load_inline` which handles torch includes
    # correctly. Skip the local pre-check for ops6k-shaped code; otherwise
    # 100% of ops6k rollouts compile_fail at the pre-check stage with
    # "fatal error: torch/extension.h: No such file or directory" and never
    # reach the real evaluator. Verified on EXP-014 job 7015618 (82/82
    # compile_fail with identical missing-header error).
    if "torch/extension.h" in code or "PYBIND11_MODULE" in code:
        return True, ""

    cu_path = None
    obj_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(code)
            cu_path = f.name

        obj_path = cu_path.replace(".cu", ".o")
        # EXP-S4 finding: WCC kernels use device lambdas (find_root etc) which
        # require --extended-lambda. Add as default so local gate matches what
        # eval_core also needs.
        proc = subprocess.run(
            ["nvcc", f"-arch={TARGET_CUDA_ARCH}", "--extended-lambda", "-c", cu_path, "-o", obj_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if proc.returncode != 0:
            return False, proc.stderr[:1000]
        return True, ""
    except FileNotFoundError:
        # nvcc not in PATH — surface this loudly instead of pretending compile
        # succeeded. EXP-004 (slurm 6874203) caught this masking eval_core
        # failures across EXP-001..EXP-003 GRPO rollouts.
        return False, "nvcc not in PATH — load CUDA module / set CUDA_HOME"
    except subprocess.TimeoutExpired:
        return False, "Local compile timed out (15s)"
    except Exception as exc:
        return False, f"Local compile raised: {type(exc).__name__}: {exc}"
    finally:
        # Cleanup must run on the exception paths too — subprocess timeouts
        # were leaking one .cu per failed rollout into $TMPDIR.
        for path in (cu_path, obj_path):
            if path is None:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass


def _compute_reward_from_result(result: dict) -> float:
    """Compute discrete milestone reward from evaluation result."""
    return compute_task_reward(result)


def _format_feedback(result: dict, reward: float, turn: int) -> str:
    """Format evaluator feedback for the next policy turn."""
    result = normalize_eval_result(result)
    parts = [f"[Turn {turn + 1} Result]"]

    if not result.get("compiles"):
        error = result.get("error", "unknown compilation error")
        parts.append(f"COMPILATION FAILED:\n{error[:MAX_ERROR_CHARS]}")
        parts.append("Fix the compilation errors above and resubmit.")
    elif not result.get("correct"):
        msg = result.get("verifier_msg") or result.get("error") or "unknown verification failure"
        parts.append(f"VERIFICATION FAILED: {msg}")
        parts.append("Your kernel produced incorrect output. Fix the implementation.")
    else:
        runtime = float(result.get("runtime_ms", 0.0) or 0.0)
        speedup_eager = float(result.get("speedup_vs_orig", 0.0) or 0.0)
        speedup_compile = float(result.get("speedup_vs_dg", 0.0) or 0.0)
        stats = result.get("runtime_stats", {}) or {}
        parts.append(f"CORRECT. Runtime: {runtime:.3f}ms")
        parts.append(f"  Speedup vs eager: {speedup_eager:.2f}x")
        if speedup_compile:
            parts.append(f"  Speedup vs torch.compile: {speedup_compile:.2f}x")
        if stats:
            parts.append(
                f"  Stats: mean={float(stats.get('mean', 0.0)):.3f}ms, "
                f"std={float(stats.get('std', 0.0)):.3f}ms"
            )

        if reward <= 1.0:
            parts.append(
                "Kernel is correct but not faster than eager PyTorch. "
                "Try reducing memory traffic or using shared memory tiling."
            )
        elif reward <= 2.0:
            parts.append(
                "Faster than eager PyTorch but not torch.compile. Push toward "
                "beating torch.compile with better occupancy or warp-level primitives."
            )

    feedback = "\n".join(parts)
    return feedback[:MAX_FEEDBACK_CHARS]


_baselines_cache: dict[str, Any] | None = None
_baselines_failure_count = 0
_BASELINE_FAILURE_LOG_EVERY = 10


def _get_baselines() -> tuple[float | None, float | None]:
    """Fetch baseline timings from eval backend (cached across calls).

    Only successful fetches are cached — a transient eval-service failure
    must not pin `{}` forever; the next call retries. Failures are logged on
    the first occurrence and then every Nth to avoid spam.
    """
    global _baselines_cache, _baselines_failure_count
    if _baselines_cache is None:
        try:
            from openenv_env.eval_backend import dispatch_eval

            _baselines_cache = dispatch_eval("profile_baselines") or {}
        except Exception as exc:
            _baselines_failure_count += 1
            if (
                _baselines_failure_count == 1
                or _baselines_failure_count % _BASELINE_FAILURE_LOG_EVERY == 0
            ):
                print(
                    f"Baseline profiling failed (attempt {_baselines_failure_count}, "
                    f"will retry on next call): {exc}"
                )
            return None, None
    return _baselines_cache.get("original_ms"), _baselines_cache.get("doublegraph_ms")


def make_multi_turn_rollout(
    max_turns: int = 3,
    skill_md_gpu: str | None = None,
    problem_metadata: list[dict] | None = None,
) -> Callable:
    """Create a task-aware rollout_func for GRPOTrainer."""
    from trl.experimental.openenv import generate_rollout_completions
    from openenv_env.skill_builder import build_skill_md

    gpu_name = skill_md_gpu or os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()
    prompt_lookup = build_prompt_lookup(problem_metadata or [])
    print(
        f"[ROLLOUT_FACTORY] make_multi_turn_rollout built rollout_func "
        f"(max_turns={max_turns}, prompts_known={len(prompt_lookup)})",
        flush=True,
    )

    def rollout_func(prompts: list[str], trainer: Any) -> dict:
        print(
            f"[ROLLOUT_CALL] rollout_func invoked with {len(prompts)} prompts",
            flush=True,
        )
        tokenizer = trainer.processing_class
        skill_context = build_skill_md(gpu_name)
        baseline_orig, baseline_dg = _get_baselines()

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_best_rewards: list[float] = []

        for prompt_idx, prompt in enumerate(prompts):
            task_row = normalize_task_row(prompt_lookup.get(prompt, {"prompt": prompt}))
            topology_ctx = format_topology_context(task_row)
            current_prompt = build_generation_prompt(
                task_row,
                skill_context=skill_context,
                topology_context=topology_ctx,
            )

            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            best_reward = -1.0

            for turn in range(max_turns):
                outputs = generate_rollout_completions(trainer, [current_prompt])[0]
                episode_prompt_ids.extend(outputs["prompt_ids"])
                episode_completion_ids.extend(outputs["completion_ids"])
                episode_logprobs.extend(outputs["logprobs"])

                completion_text = outputs.get("text") or tokenizer.decode(
                    outputs["completion_ids"], skip_special_tokens=True
                )
                code = extract_cuda_code(completion_text)
                # EXP-FIX: build result dict in every branch then route through
                # compute_task_reward() ONCE at the end. Prior code hardcoded
                # reward=-1.0 in 4 branches, bypassing reward.py entirely — that
                # made KERNELFORGE_REWARD_VERSION a no-op for compile_failed and
                # broke every prior v1/v2-shaped A/B test.
                if not code:
                    result = {
                        "compiles": False,
                        "correct": False,
                        "error": (
                            "No valid CUDA/C++ code was found. Return a fenced code block "
                            "or a raw CUDA extension source file."
                        ),
                    }
                else:
                    compiles_locally, compile_err = _local_compile_check(code)
                    if not compiles_locally:
                        result = {"compiles": False, "correct": False, "error": compile_err[:200]}
                    elif not task_row.get("supports_evaluation"):
                        result = {
                            "compiles": False,
                            "correct": False,
                            "error": task_row.get("support_reason", "Unsupported evaluation backend"),
                        }
                    else:
                        try:
                            result = evaluate_code_remote(
                                code,
                                task_row,
                                baseline_orig_ms=baseline_orig,
                                baseline_dg_ms=baseline_dg,
                            )
                        except Exception as exc:
                            print(f"  [Turn {turn + 1}] Eval dispatch failed: {exc}")
                            result = {"compiles": False, "correct": False, "error": str(exc)[:200]}
                # Canonical reward path — always re-derive via reward.py.
                # Do NOT trust a `reward` field that an eval backend may
                # have injected: it bypasses KERNELFORGE_REWARD_VERSION and
                # any future anti-hack adjustments to the reward function.
                reward = float(_compute_reward_from_result(result))

                if reward > best_reward:
                    best_reward = reward

                _append_rollout_log(
                    {
                        "prompt_index": prompt_idx,
                        "turn": turn + 1,
                        "evaluation_backend": task_row.get("evaluation_backend"),
                        "reward": reward,
                        "compiles": bool(result.get("compiles")),
                        "correct": bool(result.get("correct")),
                        "runtime_ms": float(result.get("runtime_ms", 0.0) or 0.0),
                        "speedup_vs_orig": float(result.get("speedup_vs_orig", 0.0) or 0.0),
                        "speedup_vs_dg": float(result.get("speedup_vs_dg", 0.0) or 0.0),
                    }
                )
                # EXP-007 D1 diagnostic: per-rollout one-liner to stdout
                if os.getenv("KERNELFORGE_ROLLOUT_DEBUG", "0") == "1":
                    err_str = str(result.get("error", ""))[:200].replace("\n", " ")
                    code_len = len(code) if code else 0
                    print(
                        f"[ROLLOUT prompt={prompt_idx} turn={turn + 1} reward={reward:+.2f} "
                        f"code_len={code_len} compiles={result.get('compiles')} "
                        f"correct={result.get('correct')} err='{err_str}'",
                        flush=True,
                    )
                # EXP-008 D probe: anti-hack / verifier_msg full log per rollout
                if os.getenv("KERNELFORGE_VERIFIER_DEBUG", "0") == "1":
                    vmsg = str(result.get("verifier_msg", ""))[:400].replace("\n", " ")
                    err_full = str(result.get("error", ""))[:400].replace("\n", " ")
                    print(
                        f"[VERIFIER prompt={prompt_idx} turn={turn + 1} "
                        f"compiles={result.get('compiles')} correct={result.get('correct')} "
                        f"verifier_msg='{vmsg}' error='{err_full}'",
                        flush=True,
                    )

                if reward >= 3.0 or turn == max_turns - 1:
                    break

                feedback = _format_feedback(result, reward, turn)
                current_prompt = build_generation_prompt(
                    task_row,
                    skill_context=skill_context,
                    topology_context=topology_ctx,
                ) + f"\n\n{feedback}"

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)
            all_best_rewards.append(best_reward)

            if (prompt_idx + 1) % 10 == 0 or prompt_idx == 0:
                print(
                    f"  Rollout {prompt_idx + 1}/{len(prompts)}: "
                    f"best_reward={best_reward:.3f} backend={task_row.get('evaluation_backend')}"
                )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_best_rewards,
        }

    return rollout_func


def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
    """Compute reward for each completion via the canonical extract → compile → eval → reward.py chain.

    Two modes:
    1. **rollout_func path** (multi-turn): if `env_reward` is in kwargs and matches
       len(completions), pass through — rollout_func already computed it.
    2. **TRL default-generation path** (single-turn): TRL 0.29 may not invoke
       our rollout_func in every config. In that case `env_reward` is missing
       and we MUST do the full reward computation here. **Never return a
       hardcoded -1.0 fallback** — that was the historical bypass that made
       every prior v1/v2/G/β A/B test compare two identical constant outputs.

    Inputs from TRL (per the GRPOTrainer reward_funcs contract):
      completions: list of generated text (post-tokenization decode)
      kwargs: every dataset column passed by name, e.g. `prompts`, `task_code`,
              `ops`, `evaluation_backend`, etc. — values are per-row lists
              aligned with completions.
    """
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards and len(env_rewards) == len(completions):
        return [float(r) for r in env_rewards]

    # TRL bypassed our rollout_func — compute rewards inline from completions.
    print(
        f"[REWARD_FROM_ENV] computing {len(completions)} rewards inline "
        f"(env_reward len={len(env_rewards)}); kwargs keys={sorted(kwargs.keys())}",
        flush=True,
    )

    baseline_orig, baseline_dg = _get_baselines()

    def _row_field(name: str, idx: int) -> Any:
        v = kwargs.get(name)
        if isinstance(v, list) and idx < len(v):
            return v[idx]
        return None

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        row_payload = {
            "prompt": _row_field("prompts", i) or _row_field("prompt", i),
            "task_code": _row_field("task_code", i),
            "ops": _row_field("ops", i),
            "data_source": _row_field("data_source", i),
            "evaluation_backend": _row_field("evaluation_backend", i),
            "difficulty": _row_field("difficulty", i),
        }
        # extern_c_signature drives the E1/E2 ctypes contract in the ops6k
        # evaluator — dropping it here silently degraded E2 tasks to the
        # inferred-signature fallback. Include only when the source row has it.
        extern_c_signature = _row_field("extern_c_signature", i)
        if extern_c_signature is not None:
            row_payload["extern_c_signature"] = extern_c_signature
        task_row = normalize_task_row(row_payload)

        code = extract_cuda_code(completion)
        if not code:
            result = {
                "compiles": False,
                "correct": False,
                "error": "No valid CUDA/C++ code was found.",
            }
        else:
            compiles_locally, compile_err = _local_compile_check(code)
            if not compiles_locally:
                result = {"compiles": False, "correct": False, "error": compile_err[:200]}
            elif not task_row.get("supports_evaluation"):
                result = {
                    "compiles": False,
                    "correct": False,
                    "error": task_row.get("support_reason", "Unsupported evaluation backend"),
                }
            else:
                try:
                    result = evaluate_code_remote(
                        code,
                        task_row,
                        baseline_orig_ms=baseline_orig,
                        baseline_dg_ms=baseline_dg,
                    )
                except Exception as exc:
                    print(f"[REWARD_FROM_ENV] Eval dispatch failed: {exc}", flush=True)
                    result = {"compiles": False, "correct": False, "error": str(exc)[:200]}

        # Always re-derive via reward.py — never trust a `reward` field a
        # backend may have injected. Mirrors the rollout-loop reward path.
        reward = float(_compute_reward_from_result(result))
        rewards.append(reward)

        # Diagnostic prints live on the *active* code path. TRL 0.29 does not
        # actually invoke the multi-turn rollout_func (confirmed EXP-009-B
        # 2026-05-18 log of job 6888764: [ROLLOUT_FACTORY] fired once,
        # [ROLLOUT_CALL] zero times), so the gated prints inside rollout_func
        # are dead. Keep them here so KERNELFORGE_ROLLOUT_DEBUG=1 and
        # KERNELFORGE_VERIFIER_DEBUG=1 actually surface signal.
        if os.getenv("KERNELFORGE_ROLLOUT_DEBUG", "0") == "1":
            err_str = str(result.get("error", ""))[:200].replace("\n", " ")
            code_len = len(code) if code else 0
            print(
                f"[ROLLOUT_INLINE i={i} reward={reward:+.2f} "
                f"code_len={code_len} compiles={result.get('compiles')} "
                f"correct={result.get('correct')} err='{err_str}']",
                flush=True,
            )
        if os.getenv("KERNELFORGE_VERIFIER_DEBUG", "0") == "1":
            vmsg = str(result.get("verifier_msg", ""))[:400].replace("\n", " ")
            err_full = str(result.get("error", ""))[:400].replace("\n", " ")
            sv_eager = result.get("speedup_vs_orig", 0.0)
            sv_compile = result.get("speedup_vs_dg", 0.0)
            print(
                f"[VERIFIER_INLINE i={i} compiles={result.get('compiles')} "
                f"correct={result.get('correct')} sv_eager={sv_eager} "
                f"sv_compile={sv_compile} verifier_msg='{vmsg}' "
                f"error='{err_full}']",
                flush=True,
            )
        if os.getenv("KERNELFORGE_VERIFIER_DEBUG", "0") == "1":
            import re as _re
            has_extern_c        = bool(_re.search(r'extern\s*"C"', code))
            has_wcc_kernel_decl = bool(_re.search(r'\bwcc_kernel\s*\(', code))
            has_global_wcc      = bool(_re.search(r'__global__\s+void\s+wcc_kernel', code))
            in_namespace        = bool(_re.search(r'namespace\s+\w+\s*\{', code))
            is_static_or_inline = bool(_re.search(r'\b(static|inline)\s+void\s+wcc_kernel', code))
            code_tail_truncated = bool(code) and not code.rstrip().endswith("}")
            print(
                f"[SYMPROBE_INLINE i={i} compiles={result.get('compiles')} "
                f"has_extern_c={has_extern_c} has_wcc_kernel_decl={has_wcc_kernel_decl} "
                f"has_global_wcc={has_global_wcc} in_namespace={in_namespace} "
                f"is_static_or_inline={is_static_or_inline} code_tail_truncated={code_tail_truncated}]",
                flush=True,
            )

    return rewards
