#!/usr/bin/env python3
"""Smoke test: proves core logic works without GPU/network.

Exit code 0 = all core logic verified.
Run: uv run python scripts/smoke_test.py
"""
import os
import sys

# Ensure project root is on sys.path when invoked as `python scripts/smoke_test.py`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import traceback


def main():
    errors = []
    passed = 0

    # 1. Reward computation — both v1 and v2 reward shapes (KERNELFORGE_REWARD_VERSION)
    try:
        import importlib
        import openenv_env.reward as reward_mod

        def _reload_with_version(version: str):
            os.environ["KERNELFORGE_REWARD_VERSION"] = version
            importlib.reload(reward_mod)
            return reward_mod.compute_reward, reward_mod.trloo_post_process

        # v2-shaped (current default): compiled_but_wrong → 0.0
        compute_reward, trloo_post_process = _reload_with_version("v2-shaped")
        assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0, "v2 compile fail"
        assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == 0.0, "v2 verify fail expected 0.0"
        r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
        assert abs(r - 1.0) < 1e-6, f"v2 correct, no speedup: expected 1.0, got {r}"
        r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
        assert abs(r - 2.0) < 1e-6, f"v2 2x speedup_vs_eager: expected 2.0, got {r}"
        r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=2.0)
        assert abs(r - 3.0) < 1e-6, f"v2 2x speedup_vs_compile: expected 3.0, got {r}"

        # v1-discrete-milestone (legacy, retained as switchable fallback)
        compute_reward, trloo_post_process = _reload_with_version("v1-discrete-milestone")
        assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0, "v1 compile fail"
        assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0, "v1 verify fail expected -1.0"
        r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
        assert abs(r - 1.0) < 1e-6, f"v1 correct, no speedup: expected 1.0, got {r}"

        # v3-symbol-shaped (EXP-010): 4 integrity probes per kernelforge-reward-design skill.
        compute_reward, trloo_post_process = _reload_with_version("v3-symbol-shaped")
        # Probe 1: known-good kernel must still hit the top tier
        r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=2.0)
        assert abs(r - 3.0) < 1e-6, f"v3 known-good must be 3.0, got {r}"
        # Probe 2: known-broken (compile fail) must still be -1
        r = compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0)
        assert r == -1.0, f"v3 compile-fail must be -1.0, got {r}"
        # Probe 3: symbol-missing canary must be -0.5 (NEW bucket)
        r = compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0, symbol_loaded=False)
        assert r == -0.5, f"v3 symbol-missing must be -0.5, got {r}"
        # Probe 4: no-regression — compiled + symbol OK + numerically wrong → 0.0 (matches v2)
        r = compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0, symbol_loaded=True)
        assert r == 0.0, f"v3 symbol-OK-but-wrong must be 0.0, got {r}"

        # Restore default for downstream tests in this file
        _reload_with_version("v3-symbol-shaped")
        compute_reward, trloo_post_process = reward_mod.compute_reward, reward_mod.trloo_post_process

        # TRLOO post-process: N/(N-1) scaling
        scaled = trloo_post_process([0.5, -0.3, 1.2, -0.8], n=4)
        assert abs(scaled[0] - 0.5 * 4/3) < 1e-6, "TRLOO scaling"
        print("PASS: reward.compute_reward (v1 + v2 + v3) + trloo_post_process (13 assertions)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: reward - {e}")

    # 2. GPU registry (3 GPUs)
    try:
        from openenv_env.gpu_registry import get_gpu_spec
        a100 = get_gpu_spec("a100")
        assert a100["arch"] == "sm_80"
        assert a100["sms"] == 108
        assert a100["has_tma"] is False
        h100 = get_gpu_spec("h100")
        assert h100["arch"] == "sm_90a"
        assert h100["has_tma"] is True
        b200 = get_gpu_spec("b200")
        assert b200["arch"] == "sm_100a"
        print("PASS: gpu_registry (3 GPUs verified)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: gpu_registry - {e}")

    # 3. Anti-hack (cu_flags + forbidden symbols)
    try:
        from openenv_env.anti_hack import extract_cu_flags, FORBIDDEN_SYMBOLS
        assert extract_cu_flags("") == []
        assert extract_cu_flags("// CU_FLAGS: --use_fast_math") == ["--use_fast_math"]
        assert extract_cu_flags("// CU_FLAGS: --maxrregcount=48") == ["--maxrregcount=48"]
        assert extract_cu_flags("// CU_FLAGS: --maxrregcount=256") == []  # out of range
        assert extract_cu_flags("// CU_FLAGS: --evil-flag") == []  # disallowed
        assert "torch" in FORBIDDEN_SYMBOLS
        assert "triton" in FORBIDDEN_SYMBOLS
        print("PASS: anti_hack (6 assertions)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: anti_hack - {e}")

    # 4. Cache pool (LRU eviction)
    try:
        from openenv_env.cache_pool import GPUCachePool
        pool = GPUCachePool(max_entries=2)
        pool.get_or_create("a", lambda: 1)
        pool.get_or_create("b", lambda: 2)
        pool.get_or_create("c", lambda: 3)  # evicts "a"
        assert pool.get("a") is None, "evicted entry should be gone"
        assert pool.get("c") == 3
        assert len(pool) == 2
        print("PASS: cache_pool (LRU eviction)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: cache_pool - {e}")

    # 5. Skill builder
    try:
        from openenv_env.skill_builder import build_skill_md
        md_a = build_skill_md("a100")
        assert "A100" in md_a or "sm_80" in md_a
        assert len(md_a) > 100
        md_h = build_skill_md("h100")
        assert "H100" in md_h or "TMA" in md_h
        print("PASS: skill_builder (a100 + h100)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: skill_builder - {e}")

    # 6. Curriculum manager
    try:
        from training.curriculum import CurriculumManager
        cm = CurriculumManager()
        assert cm.phase_name == "single_ops"
        p = cm.get_problem()
        assert "prompt" in p
        s = cm.status()
        assert s["phase"] == "single_ops"
        assert s["total_phases"] == 4
        print("PASS: curriculum (4 phases)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: curriculum - {e}")

    # 7. Multi-turn rollout helpers (CUDA extraction)
    try:
        from training.multi_turn_rollout import extract_cuda_code
        # Fenced code block
        cuda_block = "Here is the kernel:\n```cuda\n__global__ void add(float* a, float* b, int n) {}\n```\nDone."
        code = extract_cuda_code(cuda_block)
        assert "__global__" in code, f"Should extract fenced cuda: {code[:50]}"
        # Raw __global__
        raw = "__global__ void test(float* x) { x[0] = 1.0f; }"
        assert extract_cuda_code(raw) == raw.strip()
        # No code
        assert extract_cuda_code("Just some text without CUDA") == ""
        print("PASS: multi_turn_rollout.extract_cuda_code (3 assertions)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: multi_turn_rollout - {e}")

    # 8. PAC verification (graph invariants)
    try:
        import networkx as nx
        from verification.pac_verify import verify_wcc, edges_to_csr

        edges = [(0, 1), (1, 2), (3, 4)]
        n = 5
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        labels = {}
        for comp in nx.connected_components(G):
            root = min(comp)
            for v in comp:
                labels[v] = root

        ok, msg = verify_wcc(labels, edges, n)
        assert ok, f"PAC verification should pass: {msg}"

        # Negative test: wrong labels should fail
        bad_labels = {v: 0 for v in range(n)}  # all in one component (wrong)
        ok2, _ = verify_wcc(bad_labels, edges, n)
        assert not ok2, "Wrong labels should fail"

        print("PASS: pac_verify (correct + incorrect labels)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: pac_verify - {e}")

    # Summary
    total = passed + len(errors)
    print(f"\n{'='*50}")
    if errors:
        print(f"{len(errors)}/{total} FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"All {passed} smoke tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
