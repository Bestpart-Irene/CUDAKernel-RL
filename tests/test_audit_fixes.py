"""Regression tests for bugs found in the 2026-08-03 code audit.

Each test targets one verified defect; they fail against the pre-fix code.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest


# --- eval_model._ci_95: t-value must match the actual degrees of freedom ----

def test_ci_95_uses_correct_t_for_n2():
    from evaluation.eval_model import _ci_95

    lo, hi = _ci_95([0.0, 1.0])
    # n=2 -> df=1 -> t=12.706; se = stdev/sqrt(2) = 0.5
    assert lo == pytest.approx(0.5 - 12.706 * 0.5, rel=1e-3)
    assert hi == pytest.approx(0.5 + 12.706 * 0.5, rel=1e-3)


def test_ci_95_uses_correct_t_for_n3():
    from evaluation.eval_model import _ci_95

    vals = [0.0, 0.5, 1.0]
    import statistics

    se = statistics.stdev(vals) / math.sqrt(3)
    lo, hi = _ci_95(vals)
    # n=3 -> df=2 -> t=4.303
    assert lo == pytest.approx(0.5 - 4.303 * se, rel=1e-3)
    assert hi == pytest.approx(0.5 + 4.303 * se, rel=1e-3)


def test_ci_95_n5_unchanged():
    from evaluation.eval_model import _ci_95
    import statistics

    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    se = statistics.stdev(vals) / math.sqrt(5)
    lo, hi = _ci_95(vals)
    # n=5 -> df=4 -> t=2.776 (this was the only case the old code got right)
    assert lo == pytest.approx(3.0 - 2.776 * se, rel=1e-3)
    assert hi == pytest.approx(3.0 + 2.776 * se, rel=1e-3)


# --- eval_model seeding: multi-seed eval must actually set seeds ------------

def test_seed_everything_is_deterministic():
    from evaluation.eval_model import _seed_everything
    import random

    _seed_everything(123)
    a = random.random()
    _seed_everything(123)
    b = random.random()
    assert a == b


# --- compare_stages: hub ids must not be skipped, missing local paths must --

def test_should_skip_missing_local_checkpoint():
    from evaluation.compare_stages import _should_skip

    assert _should_skip("outputs/definitely-not-a-real-checkpoint-xyz") is True


def test_should_skip_keeps_hub_ids():
    from evaluation.compare_stages import _should_skip

    assert _should_skip("Qwen/Qwen2.5-Coder-7B-Instruct") is False


def test_should_skip_keeps_existing_local_path(tmp_path):
    from evaluation.compare_stages import _should_skip

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    assert _should_skip(str(ckpt)) is False


# --- compiler: temp workdir must not leak when output_path is provided ------

def test_compile_cuda_cleans_workdir_with_output_path(tmp_path):
    from evaluation.compiler import compile_cuda

    out = tmp_path / "kernel.so"
    result = compile_cuda(
        'extern "C" void run_kernel() {}',
        output_path=str(out),
    )
    workdir = Path(result.source_path).parent
    # nvcc is absent on dev machines: compile fails, but the temp dir must
    # still be cleaned up (that is the point of this test).
    assert not workdir.exists()


# --- task_pool: unknown task_id must fail loudly, not sample randomly -------

def test_task_pool_unknown_task_id_raises():
    from openenv_env.task_pool import TaskPool

    pool = TaskPool(tasks=[{"task_id": "a", "prompt": "p", "ops": ["relu"]}])
    with pytest.raises(KeyError):
        pool.sample(task_id="nonexistent-id")


def test_task_pool_known_task_id_still_works():
    from openenv_env.task_pool import TaskPool

    pool = TaskPool(tasks=[{"task_id": "a", "prompt": "p", "ops": ["relu"]}])
    row = pool.sample(task_id="a")
    assert row["task_id"] == "a"


# --- cache_pool: a failing factory must not destroy a healthy LRU entry -----

def test_cache_pool_failing_factory_preserves_lru():
    from openenv_env.cache_pool import GPUCachePool

    closed = []

    class Res:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    pool = GPUCachePool(max_entries=1)
    pool.get_or_create("a", lambda: Res("a"))

    def bad_factory():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        pool.get_or_create("b", bad_factory)

    # "a" must still be alive and retrievable: the factory failed, so nothing
    # should have been evicted to make room for "b".
    assert closed == []
    assert pool.get_or_create("a", lambda: Res("a2")).name == "a"
