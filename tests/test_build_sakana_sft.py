"""Regression tests for scripts/build_sakana_sft.py entrypoint renaming.

A bare whole-word `forward` -> `run_kernel` substitution mangled
`std::forward<T>(...)` (and member accesses). Only freestanding identifiers
— the definition, `&forward` references, and the pybind export string —
may be renamed.
"""
from __future__ import annotations

from scripts.build_sakana_sft import (
    _rename_freestanding,
    rename_sakana_entrypoint_to_run_kernel,
)

SNIPPET_STD_FORWARD = '''
#include <torch/extension.h>
#include <utility>

template <typename T>
torch::Tensor helper(T&& t) {
    return std::forward<T>(t);
}

torch::Tensor forward(torch::Tensor x) {
    return helper(x) * 2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "doubles the input");
}
'''


def test_std_forward_untouched_but_entrypoint_renamed():
    out = rename_sakana_entrypoint_to_run_kernel(SNIPPET_STD_FORWARD)
    # std::forward must survive untouched
    assert "std::forward<T>(t)" in out
    # ...while the definition, pybind string, and &ref are all renamed
    assert "torch::Tensor run_kernel(torch::Tensor x)" in out
    assert 'm.def("run_kernel", &run_kernel' in out
    assert "torch::Tensor forward(" not in out
    assert '"forward"' not in out


def test_member_access_forward_untouched():
    code = (
        "obj.forward(x); ptr->forward(y);\n"
        "void forward(int);\n"
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }\n'
    )
    out = rename_sakana_entrypoint_to_run_kernel(code)
    assert "obj.forward(x)" in out
    assert "ptr->forward(y)" in out
    assert "void run_kernel(int)" in out
    assert 'm.def("run_kernel", &run_kernel)' in out


def test_rename_freestanding_is_whole_word():
    # Identifiers merely containing the name must not be touched.
    out = _rename_freestanding("int forwarding = forward(1);", "forward")
    assert out == "int forwarding = run_kernel(1);"
