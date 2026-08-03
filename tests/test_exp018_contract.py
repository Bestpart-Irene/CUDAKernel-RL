"""EXP-018d: contract-contradiction regression tests.

Covers the three-way EXP-018 inconsistency found in the 2026-08-03 audit:
1. `task_interface_contract` defaulted every ops6k row to E1 while the
   evaluator independently infers E1/E2 from `get_inputs()`.
2. `_build_cuda_prompt` still demanded the forbidden pybind/torch-extension
   contract that the EXP-018 evaluator rejects at source scan.
3. The shipped dataset prompts carry the same poison (migration script).
"""
from __future__ import annotations

import json
from pathlib import Path

E1_TASK_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024)]

def get_init_inputs():
    return []
"""

E2_TASK_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, y):
        return x + y

def get_inputs():
    return [torch.randn(64, 64), torch.randn(64, 64)]

def get_init_inputs():
    return []
"""

E3_TASK_CODE = """
import torch

def get_inputs():
    return [torch.randn(8), torch.randn(8), torch.randn(8)]
"""


# --- signature inference must mirror eval_core (1 -> E1, 2 -> E2, else None)

def test_infer_signature_class_e1():
    from training.task_support import infer_signature_class

    assert infer_signature_class(E1_TASK_CODE) == "E1"


def test_infer_signature_class_e2():
    from training.task_support import infer_signature_class

    assert infer_signature_class(E2_TASK_CODE) == "E2"


def test_infer_signature_class_three_tensors_is_none():
    from training.task_support import infer_signature_class

    assert infer_signature_class(E3_TASK_CODE) is None


# --- contract text must follow the inferred class, not a hardcoded E1 -------

def test_contract_infers_e2_without_explicit_signature():
    from training.task_support import task_interface_contract

    row = {
        "prompt": "p",
        "ops": ["torch.add"],
        "task_code": E2_TASK_CODE,
        "evaluation_backend": "ops6k",
    }
    contract = task_interface_contract(row)
    assert "const float* x, const float* y" in contract


def test_contract_explicit_signature_still_wins():
    from training.task_support import task_interface_contract

    row = {
        "prompt": "p",
        "ops": ["torch.relu"],
        "task_code": E2_TASK_CODE,
        "extern_c_signature": {"class": "E1"},
        "evaluation_backend": "ops6k",
    }
    contract = task_interface_contract(row)
    assert "const float* y" not in contract


# --- dataset prompt builder must emit the extern-C contract, not pybind -----

def test_build_cuda_prompt_no_pybind_demands():
    from training.cuda_agent_integration import _build_cuda_prompt

    prompt = _build_cuda_prompt(
        {"code": E2_TASK_CODE, "ops": '["torch.add"]', "data_source": "test"}
    )
    assert prompt is not None
    # The legacy DEMAND bullet must be gone; the new contract legitimately
    # mentions torch/extension.h inside its "Do NOT include" prohibition.
    assert "- include `#include <torch/extension.h>`" not in prompt
    assert "PYBIND11_MODULE" not in prompt
    assert "Do NOT include" in prompt
    assert 'extern "C" void run_kernel(const float* x, const float* y' in prompt


# --- prompt assembly must not duplicate the contract -----------------------

def test_build_generation_prompt_no_duplicate_contract():
    from training.task_support import build_generation_prompt, task_interface_contract

    row = {
        "prompt": "task text",
        "ops": ["torch.add"],
        "task_code": E2_TASK_CODE,
        "evaluation_backend": "ops6k",
    }
    contract = task_interface_contract(row)
    row_with_contract = dict(row, prompt="task text\n\n" + contract)
    final = build_generation_prompt(row_with_contract)
    assert final.count('Evaluation contract (EXP-018a unified extern "C")') == 1


# --- dataset migration -------------------------------------------------------

def test_migrate_ops6k_prompts(tmp_path):
    from scripts.migrate_ops6k_prompts_extern_c import migrate_file

    pybind_prompt = (
        "Write a kernel.\nThe source must:\n"
        "- include `#include <torch/extension.h>`\n"
        "- export `run_kernel` via `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`\n"
    )
    rows = [
        {
            "prompt": pybind_prompt,
            "task_code": E2_TASK_CODE,
            "ops": ["torch.add"],
            "data_source": "ops_local_fallback",
            "evaluation_backend": "ops6k",
        },
        {
            "prompt": "WCC task prompt",
            "ops": ["weakly_connected_components"],
            "data_source": "doublegraph_a100",
            "evaluation_backend": "wcc",
        },
    ]
    path = tmp_path / "combined.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    stats = migrate_file(path)
    assert stats["migrated"] == 1

    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert "PYBIND11_MODULE" not in out[0]["prompt"]
    assert "- include `#include <torch/extension.h>`" not in out[0]["prompt"]
    assert 'extern "C" void run_kernel(const float* x, const float* y' in out[0]["prompt"]
    # non-ops6k rows untouched
    assert out[1]["prompt"] == "WCC task prompt"
    # backup created
    assert Path(str(path) + ".bak").exists()
    # idempotent second run
    stats2 = migrate_file(path)
    assert stats2["migrated"] == 0
