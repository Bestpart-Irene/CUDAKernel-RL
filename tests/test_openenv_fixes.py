"""Regression tests for the 2026-08-03 openenv_env bug fixes.

Covers: client abstract-method contract, baseline-cache None round-trip,
best_reward tracking on non-correct outcomes, error=None handling,
profile_baselines warning, call-time eval-backend env lookup, process-group
kill on eval timeout, gpu_registry copy semantics, and the missing
KERNELFORGE_SKILL_FILE warning.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
from unittest.mock import patch

import pytest


# --- Fix 1: KernelForgeClient must implement the EnvClient abstract hooks ----

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
StateT = TypeVar("StateT")


@dataclass
class _StepResult(Generic[ObsT]):
    """Mirror of openenv.core.client_types.StepResult."""

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False


class _RealishEnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """Minimal real ABC mirroring openenv-core 0.2.1 EnvClient.

    conftest.py stubs the SDK with MagicMock, which silently swallows the
    abstract-method contract — this stub restores enforcement so the test
    fails if any of the three hooks goes missing again.
    """

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self._ws_url = base_url

    @abstractmethod
    def _step_payload(self, action):
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload):
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload):
        raise NotImplementedError


@pytest.fixture()
def kernelforge_client_cls():
    """Import openenv_env.client against the enforcing EnvClient stub."""
    client_types_mod = types.ModuleType("openenv.core.client_types")
    client_types_mod.StepResult = _StepResult
    env_client_mod = types.ModuleType("openenv.core.env_client")
    env_client_mod.EnvClient = _RealishEnvClient

    names = ("openenv.core.client_types", "openenv.core.env_client",
             "openenv_env.client")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["openenv.core.client_types"] = client_types_mod
    sys.modules["openenv.core.env_client"] = env_client_mod
    sys.modules.pop("openenv_env.client", None)
    try:
        client_mod = importlib.import_module("openenv_env.client")
        yield client_mod.KernelForgeClient
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_client_implements_all_abstract_hooks(kernelforge_client_cls):
    """Instantiation must not raise TypeError (all abstractmethods implemented)."""
    assert not getattr(kernelforge_client_cls, "__abstractmethods__", set())
    client = kernelforge_client_cls()  # raises TypeError pre-fix
    assert client is not None


def test_client_step_payload_carries_cuda_code(kernelforge_client_cls):
    from openenv_env.models import KernelForgeAction

    client = kernelforge_client_cls()
    payload = client._step_payload(KernelForgeAction(cuda_code="__global__ void k(){}"))
    assert payload == {"cuda_code": "__global__ void k(){}"}


def test_client_parse_result_maps_server_payload(kernelforge_client_cls):
    client = kernelforge_client_cls()
    payload = {
        "observation": {
            "text": "BENCHMARK ...",
            "baseline_original_ms": 10.0,
            "baseline_doublegraph_ms": None,
            "hardware": {"arch": "sm_80"},
            "turn": 2,
            "best_reward": 2.0,
            "info": {"task_id": "t1"},
            "graph_properties": None,
            "topology_type": "power-law",
        },
        "reward": 2.0,
        "done": False,
    }
    result = client._parse_result(payload)
    assert result.reward == 2.0
    assert result.done is False
    assert result.observation.text == "BENCHMARK ..."
    assert result.observation.baseline_original_ms == 10.0
    assert result.observation.turn == 2
    assert result.observation.best_reward == 2.0
    assert result.observation.topology_type == "power-law"


def test_client_parse_state_passes_through_extras(kernelforge_client_cls):
    client = kernelforge_client_cls()
    state = client._parse_state(
        {"episode_id": "ep-1", "step_count": 3, "best_reward": 1.0}
    )
    assert state.episode_id == "ep-1"
    assert state.step_count == 3


# --- Env fixtures for fixes 2, 4, 5, 6 ---------------------------------------

_WCC_TASK = {
    "task_id": "fix_wcc",
    "prompt": "WCC task",
    "ops": ["wcc"],
    "evaluation_backend": "wcc",
}


@pytest.fixture()
def env():
    """KernelForgeEnv with a mocked eval dispatch."""
    with patch(
        "openenv_env.kernel_forge_env.KernelForgeEnv._dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"original_ms": 10.0, "doublegraph_ms": 8.0}
        from openenv_env import KernelForgeEnv

        yield KernelForgeEnv(), mock_dispatch


# --- Fix 2: baseline cache must not round-trip None -> 0.0 -> "profiled" ----

def test_cached_zero_baseline_triggers_reprofile(env):
    e, mock_dispatch = env
    from openenv_env.task_pool import TaskPool

    e.task_pool = TaskPool([dict(_WCC_TASK)])
    # Poison the cache the way the old write path did (None coerced to 0.0).
    e.task_pool.cache_baselines("fix_wcc", {"eager_ms": 0.0, "compile_ms": 0.0})

    e.reset()

    # 0.0 must be treated as "not profiled": the re-profile guard fires
    # and real baselines are fetched instead of shipping 0.0 downstream.
    assert e.original_baseline_ms == 10.0
    assert e.doublegraph_baseline_ms == 8.0


def test_step_never_caches_zero_for_missing_baseline(env):
    e, mock_dispatch = env
    from openenv_env.models import KernelForgeAction
    from openenv_env.task_pool import TaskPool

    e.task_pool = TaskPool([dict(_WCC_TASK)])
    e.reset()
    tid = (e.current_task or {}).get("task_id", "")
    assert tid == "fix_wcc"
    e.original_baseline_ms = 10.0
    e.doublegraph_baseline_ms = None  # never profiled

    mock_dispatch.return_value = {
        "compiles": True,
        "correct": True,
        "runtime_ms": 12.0,
        "runtime_stats": {"median": 12.0},
    }
    e.step(KernelForgeAction(cuda_code="__global__ void k(){}"))

    cached = e.task_pool.get_cached_baselines(tid)
    assert cached is not None
    assert cached.get("eager_ms") == 10.0
    # The missing baseline must be absent — never coerced to 0.0.
    assert "compile_ms" not in cached


# --- Fix 4: best_reward must track every outcome, not just correct kernels ---

def test_best_reward_lifts_on_verification_failure(env):
    e, mock_dispatch = env
    from openenv_env.models import KernelForgeAction

    e.reset()
    assert e.best_reward == -1.0

    mock_dispatch.return_value = {
        "compiles": True,
        "correct": False,
        "verifier_msg": "output mismatch",
    }
    obs = e.step(KernelForgeAction(cuda_code="__global__ void k(){}"))

    # v3 default: compiled + symbol OK + wrong -> 0.0, which must lift
    # best_reward above the -1.0 floor.
    assert obs.reward == 0.0
    assert e.best_reward == 0.0
    assert obs.best_reward == 0.0


# --- Fix 5: backend "error": null must not crash the compile-fail branch ----

def test_step_with_none_error_does_not_raise(env):
    e, mock_dispatch = env
    from openenv_env.models import KernelForgeAction

    e.reset()
    mock_dispatch.return_value = {"compiles": False, "error": None}
    obs = e.step(KernelForgeAction(cuda_code="bad"))

    assert obs.reward == -1.0
    assert "COMPILATION FAILED" in obs.text
    assert "Unknown error" in obs.text


# --- Fix 6: profile_baselines outages must warn (once per task id) ----------

def test_profile_baselines_failure_warns_once_per_task(env, capsys):
    e, mock_dispatch = env
    from openenv_env.task_pool import TaskPool

    e.task_pool = TaskPool([dict(_WCC_TASK)])
    mock_dispatch.side_effect = RuntimeError("eval backend down")

    e.reset()
    e.reset()  # same task id — must not warn a second time

    out = capsys.readouterr().out
    assert out.count("profile_baselines failed") == 1
    assert "eval backend down" in out


# --- Fix 7: eval backend env vars must be read at call time -----------------

def test_eval_backend_env_vars_are_live(monkeypatch):
    import openenv_env.eval_backend as eb

    monkeypatch.setenv("KERNELFORGE_EVAL_BACKEND", "modal")
    assert eb.EVAL_BACKEND == "modal"
    monkeypatch.setenv("KERNELFORGE_EVAL_BACKEND", "local")
    assert eb.EVAL_BACKEND == "local"

    monkeypatch.setenv("KERNELFORGE_EVAL_URL", "http://after-import.invalid")
    assert eb.EVAL_URL == "http://after-import.invalid"

    monkeypatch.setenv("KERNELFORGE_EVAL_SUBPROCESS_TIMEOUT", "7")
    assert eb.EVAL_SUBPROCESS_TIMEOUT == 7


def test_dispatch_eval_routes_on_call_time_backend(monkeypatch):
    import openenv_env.eval_backend as eb

    monkeypatch.setattr(eb, "_dispatch_local", lambda fn, p: {"backend": "local"})
    monkeypatch.setattr(eb, "_dispatch_modal", lambda fn, p: {"backend": "modal"})
    monkeypatch.setattr(eb, "_dispatch_http", lambda fn, p: {"backend": "http"})

    monkeypatch.setenv("KERNELFORGE_EVAL_BACKEND", "local")
    assert eb.dispatch_eval("x") == {"backend": "local"}
    monkeypatch.setenv("KERNELFORGE_EVAL_BACKEND", "modal")
    assert eb.dispatch_eval("x") == {"backend": "modal"}
    monkeypatch.setenv("KERNELFORGE_EVAL_BACKEND", "coreweave")
    assert eb.dispatch_eval("x") == {"backend": "http"}


# --- Fix 9: eval timeout must kill the whole process group ------------------

@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_kill_process_group_kills_grandchildren():
    from openenv_env.eval_backend import _kill_process_group

    proc = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 300 & echo $!; wait"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        grandchild_pid = int(proc.stdout.readline().strip())
        os.kill(grandchild_pid, 0)  # sanity: grandchild is alive

        _kill_process_group(proc, grace_s=1.0)

        assert proc.poll() is not None, "direct child still running"
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("grandchild survived the process-group kill")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# --- Fix 10: get_gpu_spec must return a copy, not the live registry ---------

def test_get_gpu_spec_mutation_does_not_poison_registry():
    from openenv_env.gpu_registry import get_gpu_spec

    spec = get_gpu_spec("a100")
    spec["sms"] = 1
    spec["features"].append("hacked")

    fresh = get_gpu_spec("a100")
    assert fresh["sms"] == 108
    assert "hacked" not in fresh["features"]


# --- Fix 11: missing KERNELFORGE_SKILL_FILE must warn, then fall through ----

def test_missing_skill_file_prints_warning(monkeypatch, capsys):
    from openenv_env.skill_builder import build_skill_md

    monkeypatch.setenv(
        "KERNELFORGE_SKILL_FILE", "definitely_missing_skill_file.md"
    )
    md = build_skill_md("h100")

    out = capsys.readouterr().out
    assert "KERNELFORGE_SKILL_FILE" in out
    assert "definitely_missing_skill_file.md" in out
    assert len(md) > 100  # fell through to static/generated SKILL.md
