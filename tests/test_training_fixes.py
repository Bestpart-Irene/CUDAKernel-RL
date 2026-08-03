"""Regression tests for the training/ fixes from the 2026-08-03 audit.

Covers the fixes whose logic is testable without torch/trl/GPU:
checkpoint gating + resume validation, GRPOConfig kwargs filtering,
extract_cuda_code fence handling, baseline-cache retry, dataset cache
rebuild policy, HF-datasets probe fallback, RFT fallback accounting,
topology count formatting, and local-compile temp cleanup.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types

import pytest

from training.checkpoint_utils import find_resumable_checkpoint, has_model_files
from training.config_utils import filter_config_kwargs


# --- fix 16: find_resumable_checkpoint ---------------------------------------


class TestFindResumableCheckpoint:
    def test_missing_dir_returns_none(self, tmp_path):
        assert find_resumable_checkpoint(str(tmp_path / "nope")) is None

    def test_empty_dir_returns_none(self, tmp_path):
        assert find_resumable_checkpoint(str(tmp_path)) is None

    def test_half_written_checkpoint_is_skipped(self, tmp_path, capsys):
        good = tmp_path / "checkpoint-10"
        good.mkdir()
        (good / "trainer_state.json").write_text("{}")
        # Newer but half-written (walltime kill before trainer_state.json)
        (tmp_path / "checkpoint-20").mkdir()

        result = find_resumable_checkpoint(str(tmp_path))
        assert result == str(good)
        assert "checkpoint-20" in capsys.readouterr().out

    def test_only_invalid_checkpoints_returns_none(self, tmp_path):
        (tmp_path / "checkpoint-5").mkdir()
        assert find_resumable_checkpoint(str(tmp_path)) is None

    def test_numeric_ordering_not_lexicographic(self, tmp_path):
        for step in (9, 100):
            ckpt = tmp_path / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "trainer_state.json").write_text("{}")
        assert find_resumable_checkpoint(str(tmp_path)).endswith("checkpoint-100")

    def test_non_checkpoint_dirs_ignored(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "checkpoint-final").mkdir()
        assert find_resumable_checkpoint(str(tmp_path)) is None


# --- fixes 4 + 5: stage handoff gate accepts adapter-only checkpoints --------


class TestHasModelFiles:
    def test_full_model_dir(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert has_model_files(str(tmp_path)) is True

    def test_peft_adapter_only_dir(self, tmp_path):
        # What stage1 actually saves: adapter_config.json, no config.json.
        (tmp_path / "adapter_config.json").write_text("{}")
        assert has_model_files(str(tmp_path)) is True

    def test_bare_dir_is_not_a_checkpoint(self, tmp_path):
        assert has_model_files(str(tmp_path)) is False

    def test_missing_dir(self, tmp_path):
        assert has_model_files(str(tmp_path / "nope")) is False


# --- fix 2: filter kwargs against the installed config class -----------------


@dataclasses.dataclass
class _FakeConfig:
    output_dir: str = "out"
    learning_rate: float = 1e-5


class TestFilterConfigKwargs:
    def test_drops_unknown_keys_with_warning(self, capsys):
        kwargs = {"output_dir": "x", "max_prompt_length": 3072}
        kept = filter_config_kwargs(_FakeConfig, kwargs)
        assert kept == {"output_dir": "x"}
        out = capsys.readouterr().out
        assert "max_prompt_length" in out
        assert "_FakeConfig" in out

    def test_passes_known_keys_silently(self, capsys):
        kwargs = {"output_dir": "x", "learning_rate": 3e-6}
        assert filter_config_kwargs(_FakeConfig, kwargs) == kwargs
        assert capsys.readouterr().out == ""

    def test_var_keyword_class_passes_through(self):
        class Anything:
            def __init__(self, **kwargs):
                pass

        kwargs = {"whatever": 1}
        assert filter_config_kwargs(Anything, kwargs) == kwargs

    def test_plain_class_signature_filtering(self):
        class Plain:
            def __init__(self, a, b=2):
                pass

        assert filter_config_kwargs(Plain, {"a": 1, "c": 3}) == {"a": 1}


# --- fix 8: extract_cuda_code fence handling ----------------------------------


class TestExtractCudaCodeFences:
    def test_tagged_fence_unchanged(self):
        from training.multi_turn_rollout import extract_cuda_code

        text = "Here:\n```cuda\n__global__ void add(float* a) {}\n```\nDone."
        code = extract_cuda_code(text)
        assert code == "__global__ void add(float* a) {}"

    def test_untagged_fence_with_global_kernel(self):
        from training.multi_turn_rollout import extract_cuda_code

        text = "Sure:\n```\n__global__ void relu(float* x, int n) {}\n```\nEnjoy."
        code = extract_cuda_code(text)
        assert code == "__global__ void relu(float* x, int n) {}"
        assert "```" not in code

    def test_untagged_truncated_fence(self):
        from training.multi_turn_rollout import extract_cuda_code

        text = '```\nextern "C" void run_kernel(const float* x, float* out, int n) {'
        code = extract_cuda_code(text)
        assert code.startswith('extern "C" void run_kernel')
        assert "```" not in code

    def test_no_fence_raw_kernel(self):
        from training.multi_turn_rollout import extract_cuda_code

        raw = "__global__ void k(int* d) { d[0] = 1; }"
        assert extract_cuda_code(raw) == raw

    def test_raw_fallback_strips_stray_fence_lines(self):
        from training.multi_turn_rollout import extract_cuda_code

        text = "```text\nprose only, no kernel\n```\n__global__ void k(int* d) {}"
        code = extract_cuda_code(text)
        assert "```" not in code
        assert "__global__ void k" in code

    def test_untagged_fence_without_code_is_not_extracted(self):
        from training.multi_turn_rollout import extract_cuda_code

        assert extract_cuda_code("```\njust prose here\n```") == ""

    def test_no_code_at_all(self):
        from training.multi_turn_rollout import extract_cuda_code

        assert extract_cuda_code("no kernels anywhere") == ""


# --- fix 7: baseline cache must not pin a transient failure -------------------


class TestBaselinesRetry:
    def _reset(self, mtr):
        mtr._baselines_cache = None
        mtr._baselines_failure_count = 0

    def test_failure_is_not_cached_then_success_is(self, monkeypatch, capsys):
        import training.multi_turn_rollout as mtr
        import openenv_env.eval_backend as eb

        self._reset(mtr)

        def boom(*args, **kwargs):
            raise ConnectionError("eval service down")

        monkeypatch.setattr(eb, "dispatch_eval", boom)
        assert mtr._get_baselines() == (None, None)
        assert mtr._baselines_cache is None  # NOT pinned to {}
        assert "will retry" in capsys.readouterr().out

        monkeypatch.setattr(
            eb, "dispatch_eval",
            lambda *a, **k: {"original_ms": 2.5, "doublegraph_ms": 1.5},
        )
        assert mtr._get_baselines() == (2.5, 1.5)
        assert mtr._baselines_cache == {"original_ms": 2.5, "doublegraph_ms": 1.5}
        self._reset(mtr)

    def test_failures_log_first_and_every_nth(self, monkeypatch, capsys):
        import training.multi_turn_rollout as mtr
        import openenv_env.eval_backend as eb

        self._reset(mtr)
        monkeypatch.setattr(
            eb, "dispatch_eval",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")),
        )
        for _ in range(mtr._BASELINE_FAILURE_LOG_EVERY):
            mtr._get_baselines()
        logged = [
            line for line in capsys.readouterr().out.splitlines()
            if "Baseline profiling failed" in line
        ]
        assert len(logged) == 2  # first + Nth, not every call
        self._reset(mtr)


# --- fix 12: _local_compile_check must clean temp files on exceptions ---------


class TestLocalCompileCleanup:
    def test_timeout_does_not_leak_cu_file(self, tmp_path, monkeypatch):
        import training.multi_turn_rollout as mtr

        monkeypatch.setattr(mtr, "LOCAL_COMPILE_CHECK", True)
        real_ntf = mtr.tempfile.NamedTemporaryFile

        def ntf_in_tmp(*args, **kwargs):
            kwargs["dir"] = str(tmp_path)
            return real_ntf(*args, **kwargs)

        monkeypatch.setattr(mtr.tempfile, "NamedTemporaryFile", ntf_in_tmp)

        def timeout_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="nvcc", timeout=15)

        monkeypatch.setattr(mtr.subprocess, "run", timeout_run)

        ok, msg = mtr._local_compile_check("__global__ void k() {}")
        assert ok is False
        assert "timed out" in msg
        assert list(tmp_path.iterdir()) == []


# --- fix 14: inline reward path must forward extern_c_signature ---------------


class TestInlineRewardExternCSignature:
    def test_extern_c_signature_reaches_evaluator(self, monkeypatch):
        import training.multi_turn_rollout as mtr

        monkeypatch.setattr(mtr, "LOCAL_COMPILE_CHECK", False)
        monkeypatch.setattr(mtr, "_baselines_cache", {})

        captured: list[dict] = []

        def fake_eval(code, task_row, **kwargs):
            captured.append(task_row)
            return {"compiles": True, "correct": True, "speedup_vs_orig": 1.0}

        monkeypatch.setattr(mtr, "evaluate_code_remote", fake_eval)

        task_code = (
            "import torch\n"
            "def get_inputs():\n    return [torch.randn(4)]\n"
            "def get_init_inputs():\n    return []\n"
        )
        completions = ["```cuda\n__global__ void k(float* x) {}\n```"]
        rewards = mtr.reward_from_env(
            completions,
            prompts=["p"],
            task_code=[task_code],
            ops=[["relu"]],
            data_source=["ops_6k"],
            difficulty=[1],
            extern_c_signature=[{"class": "E2"}],
        )
        assert len(rewards) == 1
        assert captured, "evaluator was never dispatched"
        assert captured[0]["extern_c_signature"] == {"class": "E2"}


# --- fix 9: HF datasets probe must fall back on a shadow namespace package ----


class TestHfDatasetsProbe:
    def test_shadow_module_without_dataset_attr_falls_back(self):
        from training.dataset_loader import _probe_hf_datasets

        original = sys.modules.get("datasets")
        shadow = types.ModuleType("datasets")  # no Dataset attribute
        sys.modules["datasets"] = shadow
        try:
            assert _probe_hf_datasets() is None
            # The bogus cache entry must be dropped so later probes can retry.
            assert sys.modules.get("datasets") is not shadow
        finally:
            if original is not None:
                sys.modules["datasets"] = original
            else:
                sys.modules.pop("datasets", None)

    def test_real_looking_module_is_returned(self):
        from training.dataset_loader import _probe_hf_datasets

        original = sys.modules.get("datasets")
        fake = types.ModuleType("datasets")
        fake.Dataset = object
        sys.modules["datasets"] = fake
        try:
            assert _probe_hf_datasets() is fake
        finally:
            if original is not None:
                sys.modules["datasets"] = original
            else:
                sys.modules.pop("datasets", None)


# --- fix 10: combined dataset cache must never be rebuilt implicitly ----------


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestCombinedDatasetRebuildPolicy:
    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("KERNELFORGE_ALLOW_DATASET_REBUILD", raising=False)

    def _spy_builder(self, monkeypatch, rows):
        import training.dataset_loader as dl

        calls = {"build": 0, "write": 0}

        def fake_build(**kwargs):
            calls["build"] += 1
            return rows

        def fake_write(rows_, path):
            calls["write"] += 1

        monkeypatch.setattr(dl, "build_combined_dataset", fake_build)
        monkeypatch.setattr(dl, "write_jsonl", fake_write)
        return calls

    def test_consistent_cache_served_without_rebuild(self, tmp_path, monkeypatch):
        import training.dataset_loader as dl

        cache = tmp_path / "combined.jsonl"
        rows = [{"prompt": "p", "task_code": "code"}]
        _write_rows(cache, rows)
        calls = self._spy_builder(monkeypatch, [])

        out = dl._load_or_build_combined_rows("m", 8, 42, str(cache))
        assert out == rows
        assert calls == {"build": 0, "write": 0}

    def test_inconsistent_cache_warns_and_serves_stale(self, tmp_path, monkeypatch, capsys):
        import training.dataset_loader as dl

        cache = tmp_path / "combined.jsonl"
        # Manually curated rows with NO task_code — the old heuristic
        # destroyed these by rebuilding over them.
        rows = [{"prompt": "vector_add_e2", "task_code": None}]
        _write_rows(cache, rows)
        calls = self._spy_builder(monkeypatch, [{"prompt": "rebuilt"}])

        out = dl._load_or_build_combined_rows("m", 8, 42, str(cache))
        assert out == rows
        assert calls == {"build": 0, "write": 0}
        assert "KERNELFORGE_ALLOW_DATASET_REBUILD" in capsys.readouterr().out

    def test_inconsistent_cache_rebuilds_when_env_set(self, tmp_path, monkeypatch):
        import training.dataset_loader as dl

        monkeypatch.setenv("KERNELFORGE_ALLOW_DATASET_REBUILD", "1")
        cache = tmp_path / "combined.jsonl"
        _write_rows(cache, [{"prompt": "old", "task_code": None}])
        rebuilt = [{"prompt": "rebuilt", "task_code": "code"}]
        calls = self._spy_builder(monkeypatch, rebuilt)

        out = dl._load_or_build_combined_rows("m", 8, 42, str(cache))
        assert out == rebuilt
        assert calls == {"build": 1, "write": 1}

    def test_missing_cache_without_env_raises(self, tmp_path, monkeypatch):
        import training.dataset_loader as dl

        calls = self._spy_builder(monkeypatch, [{"prompt": "x"}])
        with pytest.raises(RuntimeError, match="KERNELFORGE_ALLOW_DATASET_REBUILD"):
            dl._load_or_build_combined_rows("m", 8, 42, str(tmp_path / "absent.jsonl"))
        assert calls == {"build": 0, "write": 0}

    def test_missing_cache_with_env_builds(self, tmp_path, monkeypatch):
        import training.dataset_loader as dl

        monkeypatch.setenv("KERNELFORGE_ALLOW_DATASET_REBUILD", "1")
        built = [{"prompt": "fresh", "task_code": "code"}]
        calls = self._spy_builder(monkeypatch, built)

        out = dl._load_or_build_combined_rows("m", 8, 42, str(tmp_path / "absent.jsonl"))
        assert out == built
        assert calls == {"build": 1, "write": 1}


# --- fix 6: RFT generation fallbacks must be counted and all-fallback fatal ---


class TestRftFallbackAccounting:
    def _collector(self):
        from training.rft_filter import TrajectoryCollector

        return TrajectoryCollector(model_path="outputs/does-not-exist")

    def test_fallback_is_counted_per_attempt(self, monkeypatch):
        collector = self._collector()
        monkeypatch.setattr(
            collector, "_get_generator",
            lambda: (_ for _ in ()).throw(RuntimeError("no checkpoint")),
        )
        out = collector._get_model_response("prompt")
        assert "__global__" in out  # fallback template
        assert collector._generation_attempts == 1
        assert collector._generation_fallbacks == 1
        collector._get_model_response("prompt")
        assert collector._generation_fallbacks == 2

    def test_all_fallbacks_raises_runtime_error(self, monkeypatch):
        import training.rft_filter as rf

        collector = self._collector()
        task = {"prompt": "write a WCC kernel", "ops": ["wcc"]}
        monkeypatch.setattr(collector, "_get_task_pool", lambda: [task])
        monkeypatch.setattr(
            collector, "_get_generator",
            lambda: (_ for _ in ()).throw(RuntimeError("no checkpoint")),
        )
        monkeypatch.setattr(
            rf, "evaluate_code_remote",
            lambda code, t, **kw: {"reward": 1.0, "compiles": True, "correct": True},
        )
        with pytest.raises(RuntimeError, match="fell back"):
            collector.collect_trajectories(num_trajectories=3)

    def test_successful_generation_does_not_raise(self, monkeypatch):
        import training.rft_filter as rf

        collector = self._collector()
        task = {"prompt": "write a WCC kernel", "ops": ["wcc"]}
        monkeypatch.setattr(collector, "_get_task_pool", lambda: [task])

        def fake_generator(prompt, **kwargs):
            return [{"generated_text": prompt + "\n```cuda\n__global__ void k() {}\n```"}]

        monkeypatch.setattr(collector, "_get_generator", lambda: fake_generator)
        monkeypatch.setattr(
            rf, "evaluate_code_remote",
            lambda code, t, **kw: {"reward": 1.0, "compiles": True, "correct": True},
        )
        trajectories = collector.collect_trajectories(num_trajectories=2)
        assert len(trajectories) == 2
        assert collector._generation_fallbacks == 0


# --- fix 11: topology counts must format ints and pass '?' through ------------


class TestTopologyCountFormatting:
    def test_fmt_count_int(self):
        from training.curriculum import _fmt_count

        assert _fmt_count(1234567) == "1,234,567"

    def test_fmt_count_string_passthrough(self):
        from training.curriculum import _fmt_count

        assert _fmt_count("?") == "?"

    def test_topology_context_missing_counts_does_not_raise(self):
        from training.curriculum import format_topology_context

        ctx = format_topology_context({"graph_properties": {"type": "power-law"}})
        assert "Vertices: ?" in ctx

    def test_topology_context_formats_counts(self):
        from training.curriculum import format_topology_context

        ctx = format_topology_context(
            {"graph_properties": {"num_vertices": 100000, "num_edges": 1500000}}
        )
        assert "100,000" in ctx
        assert "1,500,000" in ctx
