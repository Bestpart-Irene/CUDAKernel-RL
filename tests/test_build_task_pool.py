"""Regression tests for tasks/build_task_pool.py task filtering.

The DENY_OPS scan must guard the kernel computation (Model.forward), not
input generation — get_inputs() legitimately uses torch.rand/torch.randn in
essentially every Ops-6K task, and scanning the whole file rejected 100%
of real tasks.
"""
from __future__ import annotations

from tasks.build_task_pool import (
    _difficulty_from_ops,
    _strip_input_generators,
    is_stateless_evaluable,
)

TASK_RANDN_INPUTS_ONLY = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.randn(1024), torch.randn(1024)]


def get_init_inputs():
    return []
'''

TASK_RANDPERM_IN_FORWARD = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        idx = torch.randperm(x.shape[0])
        return x[idx]


def get_inputs():
    return [torch.randn(1024)]


def get_init_inputs():
    return []
'''


def test_randn_in_get_inputs_is_allowed():
    assert is_stateless_evaluable(TASK_RANDN_INPUTS_ONLY)


def test_randperm_in_model_forward_is_denied():
    assert not is_stateless_evaluable(TASK_RANDPERM_IN_FORWARD)


def test_strip_input_generators_removes_only_input_fns():
    stripped = _strip_input_generators(TASK_RANDN_INPUTS_ONLY)
    assert "get_inputs" not in stripped
    assert "get_init_inputs" not in stripped
    assert "class Model" in stripped
    assert "torch.randn" not in stripped

    # Nondeterminism inside the kernel computation must survive stripping
    # so the deny scan still sees it.
    stripped2 = _strip_input_generators(TASK_RANDPERM_IN_FORWARD)
    assert "torch.randperm" in stripped2


def test_strip_input_generators_passes_through_unparseable_code():
    broken = "def get_inputs(:\n    return ["
    assert _strip_input_generators(broken) == broken


def test_difficulty_from_ops_mirrors_combined_dataset_heuristic():
    assert _difficulty_from_ops([]) == 1
    assert _difficulty_from_ops(["relu"]) == 1
    assert _difficulty_from_ops(["mul", "add"]) == 2
    assert _difficulty_from_ops(["mul", "add", "exp"]) == 3
