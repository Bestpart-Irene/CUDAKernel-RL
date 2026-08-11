"""Anti-reward-hacking utilities for kernel evaluation.

Extends beyond forbidden-symbol scanning with runtime anti-hack checks
inspired by Dr. Kernel's three failure modes:
  1. Reward hacking (decoy kernels, no-ops)
  2. Lazy optimization (passthrough, constant output)
  3. Output shape violations

See: https://github.com/hkust-nlp/KernelGYM/blob/main/drkernel/README.md
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any

ALLOWED_CU_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
}
ALLOWED_CU_FLAG_PREFIXES = ("--maxrregcount=",)

FORBIDDEN_SYMBOLS = [
    "torch",
    "at::Tensor",
    "c10::",
    "torch::autograd",
    "triton",
    "torch.compile",
    "torch.nn.functional",
]

# Source-level patterns that reject a candidate before it even reaches nvcc.
# These exist because every PyTorch C++ extension links libtorch and therefore
# always shows torch::/at::/c10:: symbols in `nm -D` — so post-link symbol
# scanning is useless on the ops6k path. Source-level rejection is the only
# mechanism that closes the EXP-015'-A torch::-API delegation channel without
# breaking the legitimate WCC extern "C" path. Pattern list deliberately
# matches CUDA-Agent §13 anti-hack stack layer 1 ("no torch headers in the
# .cu file").
FORBIDDEN_SOURCE_PATTERNS = [
    r'#\s*include\s*[<"]torch/',
    r'#\s*include\s*[<"]ATen/',
    r'#\s*include\s*[<"]c10/',
    r'\btorch::',
    r'\bat::',
    r'\bc10::',
]

# Host-library API namespaces used by scripts/deep_hack_scan.py's audit grep
# (EXP-018c criterion 3). Deliberately NOT wired into the eval-path scan:
# thrust:: was never part of FORBIDDEN_SOURCE_PATTERNS, and EXP-019 is an
# integrity restoration, not a reward-semantics change.
FORBIDDEN_API_NAMESPACES = [
    r"\btorch::",
    r"\bat::",
    r"\bc10::",
    r"\bthrust::",
]

# ---------------------------------------------------------------------------
# EXP-019 fix (d): macro-indirect evasion detector for scan_source_forbidden.
#
# The raw-text regex scan can be evaded by preprocessor indirection: computed
# includes (#define X <torch/extension.h> + #include X), object-macro aliases
# (#define NS torch), token pasting (tor##ch), line splicing (tor\<newline>ch)
# and universal-character-name identifiers (torch). The helpers below
# normalize the source the way cpp would (splice -> comments-to-space -> UCN
# decode -> ## paste) and expand object-like #defines to a fixed point, then
# the same FORBIDDEN_SOURCE_PATTERNS are rescanned. Residual limits are
# documented in research/experiments/EXP-019-anti-hack-integrity.md.
# ---------------------------------------------------------------------------

_LINE_SPLICE_RE = re.compile(r"\\\r?\n")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_UCN_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")
_TOKEN_PASTE_RE = re.compile(r"\s*##\s*")
# #include whose next token is neither <...> nor "..." — a computed include.
_COMPUTED_INCLUDE_RE = re.compile(r'#\s*include\s+(?![<"])[A-Za-z_]')
# Object-like macro: '#define NAME body' where NAME is NOT immediately
# followed by '(' (that would make it function-like).
_OBJECT_MACRO_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(?!\()[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)
_MACRO_EXPANSION_MAX_ITERS = 8
_MACRO_EXPANSION_SIZE_FLOOR = 256 * 1024  # bytes


def _decode_ucn(match: re.Match[str]) -> str:
    hex_digits = match.group(1) or match.group(2)
    try:
        return chr(int(hex_digits, 16))
    except (ValueError, OverflowError):
        return match.group(0)


def _normalize_source(source: str) -> str:
    """Approximate cpp's early translation phases for detection purposes.

    Order matters and mirrors the compiler: splice backslash-newlines, replace
    each comment with a single space (so `tor/**/ch` correctly stays two
    tokens), decode UCN escapes, then collapse `##` so pasted tokens join
    (`tor ## /*x*/ ch` -> `torch`, since comments already became spaces).
    """
    text = _LINE_SPLICE_RE.sub("", source)
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(" ", text)
    text = _UCN_RE.sub(_decode_ucn, text)
    text = _TOKEN_PASTE_RE.sub("", text)
    return text


def _expand_object_macros(text: str) -> str | None:
    """Fixed-point expansion of object-like #define macros.

    Returns the expanded text, or None when expansion blows up past
    max(4x input, 256 KiB) — self-referential/exponential macro games have no
    legitimate place in a generated kernel, so the caller fails closed.
    """
    macros = {name: body for name, body in _OBJECT_MACRO_RE.findall(text)}
    if not macros:
        return text

    size_cap = max(4 * len(text), _MACRO_EXPANSION_SIZE_FLOOR)
    for _ in range(_MACRO_EXPANSION_MAX_ITERS):
        changed = False
        for name, body in macros.items():
            # Lookbehind blocks '#NAME' (stringize) and partial-word matches.
            # The macro's own '#define NAME' line may get rewritten too; that
            # is harmless for detection because the dict was captured up front.
            pattern = re.compile(r"(?<![\w#])" + re.escape(name) + r"\b")
            new_text, n = pattern.subn(body.replace("\\", "\\\\"), text)
            if n:
                text = new_text
                changed = True
            if len(text) > size_cap:
                return None
        if not changed:
            break
    return text


def _scan_patterns(text: str, note: str = "") -> str | None:
    suffix = f" ({note})" if note else ""
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return (
                f"forbidden source pattern{suffix}: {pat!r} matched {m.group(0)!r}"
            )
    return None


def scan_source_forbidden(source: str) -> str | None:
    """Return rejection reason if source matches a forbidden pattern, else None.

    Source-level pre-compile check for the unified extern "C" eval contract
    (EXP-018a). Closes the torch::-API delegation reward-hack channel that
    invalidated EXP-015'-A. Must run BEFORE nvcc to keep `nm -D` scans
    meaningful (no libtorch linkage means no false positives).

    EXP-019 fix (d) layers a macro-indirect detector on top of the original
    raw-text scan (pass 0, unchanged):
      D1  reject the token-pasting operator '##' outright (no legitimate use
          under the contract; covers function-like paste macros a regex-level
          expander cannot simulate),
      D2  reject computed includes ('#include SOME_MACRO'),
      D3  rescan the same patterns on cpp-normalized text (line splice,
          comments->space, UCN decode, '##' paste),
      D4  rescan after object-like #define fixed-point expansion,
      D5  fail closed on macro-expansion blowup.
    """
    # Pass 0 — original raw-text scan (byte-for-byte the pre-EXP-019 check).
    reason = _scan_patterns(source)
    if reason:
        return reason

    # D3 groundwork — normalized view of the source.
    spliced = _LINE_SPLICE_RE.sub("", source)
    decommented = _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", spliced))

    # D1 — token pasting has no legitimate use in a contract kernel.
    if "##" in decommented:
        return (
            "macro-indirect: token-pasting operator '##' is not permitted "
            "under the extern \"C\" eval contract"
        )

    # D2 — computed includes (#include MACRO) are rejected outright.
    m = _COMPUTED_INCLUDE_RE.search(decommented)
    if m:
        return f"macro-indirect: computed #include is not permitted: {m.group(0)!r}"

    # D3 — rescan on normalized text (splice/comment/UCN/paste applied).
    normalized = _normalize_source(source)
    reason = _scan_patterns(normalized, note="macro-indirect")
    if reason:
        return reason

    # D4/D5 — object-like macro expansion to a fixed point, then rescan.
    expanded = _expand_object_macros(normalized)
    if expanded is None:
        return (
            "macro-indirect: object-like macro expansion exceeded the size cap "
            "— suspicious self-referential/exponential macro definitions"
        )
    return _scan_patterns(expanded, note="macro-indirect")


def extract_cu_flags(cuda_code: str) -> list[str]:
    """Extract whitelisted ``// CU_FLAGS: ...`` tokens from CUDA source."""
    extracted: list[str] = []
    for line in cuda_code.splitlines():
        stripped = line.strip()
        if not stripped.startswith("// CU_FLAGS:"):
            continue

        for token in stripped.replace("// CU_FLAGS:", "").strip().split():
            if token in ALLOWED_CU_FLAGS:
                extracted.append(token)
                continue

            if any(token.startswith(prefix) for prefix in ALLOWED_CU_FLAG_PREFIXES):
                try:
                    value = int(token.split("=", 1)[1])
                except (IndexError, ValueError):
                    continue
                if 16 <= value <= 128:
                    extracted.append(token)

    # preserve order, drop duplicates
    return list(dict.fromkeys(extracted))


def scan_forbidden_symbols(so_path: str | Path) -> str | None:
    """Return a failure reason if forbidden dynamic symbols are detected.

    EXP-019 fix (c): this scan FAILS CLOSED. An nm failure (nm missing, the
    binary missing/unreadable, timeout, nonzero exit) used to return None —
    indistinguishable from a clean scan, so a broken detection environment
    silently passed every candidate. Any scan failure now returns a
    'hack_suspected' verdict string with the underlying error recorded; every
    caller already treats a non-None return as a failed verdict.
    """
    so_path = str(so_path)
    try:
        proc = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # nm missing, timeout, OS error — fail closed
        return (
            f"hack_suspected: symbol scan failed (nm could not run: {exc!r}) "
            "— failing closed"
        )

    if proc.returncode != 0:
        stderr_head = (proc.stderr or "").strip()[:200]
        return (
            f"hack_suspected: symbol scan failed (nm exit {proc.returncode} "
            f"on {so_path!r}: {stderr_head or 'no stderr'}) — failing closed"
        )

    symbols = proc.stdout
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in symbols:
            return f"Forbidden symbol detected: {forbidden}"
    return None


# ---------------------------------------------------------------------------
# Runtime anti-hack checks (Dr. Kernel-inspired)
# ---------------------------------------------------------------------------


def check_output_not_constant(
    output1: Any,
    output2: Any,
) -> tuple[bool, str]:
    """Verify outputs differ when inputs differ.

    Catches decoy kernels that return a hardcoded constant regardless of input.
    Run the candidate kernel with two different random inputs — if the outputs
    are identical, the kernel is likely ignoring the input.

    Returns (passed, reason).
    """
    import torch

    def _flatten(val: Any) -> list:
        if isinstance(val, torch.Tensor):
            return [val]
        if isinstance(val, (list, tuple)):
            out = []
            for item in val:
                out.extend(_flatten(item))
            return out
        if isinstance(val, dict):
            out = []
            for item in val.values():
                out.extend(_flatten(item))
            return out
        return []

    tensors1 = _flatten(output1)
    tensors2 = _flatten(output2)

    if len(tensors1) != len(tensors2):
        return True, "Different output structure"

    all_identical = True
    for t1, t2 in zip(tensors1, tensors2):
        if t1.shape != t2.shape:
            return True, "Shapes differ"
        if not torch.equal(t1, t2):
            all_identical = False
            break

    if all_identical and len(tensors1) > 0:
        return False, "Output is constant across different inputs (likely decoy kernel)"

    return True, "Outputs differ for different inputs"


# EXP-019 fix (b): explicit unit conversion for the no-op time floor.
# The runtime fed to check_not_noop is torch.cuda.Event.elapsed_time() /
# cupy.cuda.get_elapsed_time() output — MILLISECONDS. A CUDA kernel *launch
# alone* costs ~2μs, so the old threshold of 0.001 ms (1μs) sat BELOW the
# physical floor of the timing path and could never fire (dead check).
_US_PER_MS = 1000.0
_MIN_PLAUSIBLE_KERNEL_TIME_US = 2.0  # kernel-launch overhead floor, microseconds
_MIN_PLAUSIBLE_KERNEL_TIME_MS = _MIN_PLAUSIBLE_KERNEL_TIME_US / _US_PER_MS  # 0.002 ms


def check_not_noop(runtime_ms: float) -> tuple[bool, str]:
    """Flag suspiciously fast kernels that likely skip computation.

    Any real CUDA kernel launch takes at least ~2μs (0.002 ms) of launch
    overhead, and CUDA-event timing cannot read meaningfully below that for
    any code path that actually launches work. A median runtime below the
    2μs floor therefore means no kernel ran (or timing was subverted).
    Non-finite, zero, and negative readings fail closed for the same reason:
    they are only producible by a broken or gamed timing path.

    Args:
        runtime_ms: Median kernel runtime in MILLISECONDS
            (torch.cuda.Event.elapsed_time units).

    Returns (passed, reason).
    """
    try:
        runtime_ms = float(runtime_ms)
    except (TypeError, ValueError):
        return False, (
            f"Runtime {runtime_ms!r} is not a number — timing integrity failure"
        )
    if not math.isfinite(runtime_ms):
        return False, (
            f"Runtime {runtime_ms!r}ms is not finite — timing integrity failure"
        )
    if runtime_ms <= 0.0:
        return False, (
            f"Runtime {runtime_ms:.6f}ms is zero or negative — "
            "timing integrity failure"
        )
    runtime_us = runtime_ms * _US_PER_MS
    if runtime_us < _MIN_PLAUSIBLE_KERNEL_TIME_US:
        return False, (
            f"Runtime {runtime_us:.3f}μs ({runtime_ms:.6f}ms) is below the "
            f"{_MIN_PLAUSIBLE_KERNEL_TIME_US:.0f}μs kernel-launch floor — "
            "likely a no-op kernel"
        )
    return True, f"Runtime {runtime_ms:.3f}ms is plausible"


def check_not_passthrough(
    output: Any,
    inputs: list[Any],
) -> tuple[bool, str]:
    """Verify output is not identical to any input tensor.

    Catches lazy optimizations where the kernel just returns the input unchanged.

    Returns (passed, reason).
    """
    import torch

    def _get_first_tensor(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            return val
        if isinstance(val, (list, tuple)):
            for item in val:
                t = _get_first_tensor(item)
                if t is not None:
                    return t
        if isinstance(val, dict):
            for item in val.values():
                t = _get_first_tensor(item)
                if t is not None:
                    return t
        return None

    out_tensor = _get_first_tensor(output)
    if out_tensor is None:
        return True, "No tensor output to check"

    for i, inp in enumerate(inputs):
        inp_tensor = _get_first_tensor(inp) if not isinstance(inp, torch.Tensor) else inp
        if inp_tensor is None:
            continue
        if inp_tensor.shape == out_tensor.shape and torch.equal(inp_tensor, out_tensor):
            return False, f"Output is identical to input[{i}] — likely passthrough"

    return True, "Output differs from all inputs"


def check_shapes_match(
    candidate_output: Any,
    reference_output: Any,
) -> tuple[bool, str]:
    """Verify candidate output shapes match reference output shapes.

    Returns (passed, reason).
    """
    import torch

    def _get_shapes(val: Any) -> list[tuple[int, ...]]:
        if isinstance(val, torch.Tensor):
            return [tuple(val.shape)]
        if isinstance(val, (list, tuple)):
            out = []
            for item in val:
                out.extend(_get_shapes(item))
            return out
        if isinstance(val, dict):
            out = []
            for item in val.values():
                out.extend(_get_shapes(item))
            return out
        return []

    cand_shapes = _get_shapes(candidate_output)
    ref_shapes = _get_shapes(reference_output)

    if len(cand_shapes) != len(ref_shapes):
        return False, (
            f"Output tensor count mismatch: candidate={len(cand_shapes)}, "
            f"reference={len(ref_shapes)}"
        )

    for i, (cs, rs) in enumerate(zip(cand_shapes, ref_shapes)):
        if cs != rs:
            return False, f"Shape mismatch at output[{i}]: candidate={cs}, reference={rs}"

    return True, f"All {len(cand_shapes)} output shapes match"


def run_anti_hack_suite(
    candidate_outputs: list[Any],
    reference_outputs: list[Any],
    inputs_list: list[list[Any]],
    runtime_ms: float,
) -> tuple[bool, str]:
    """Run the full anti-hack check suite.

    Args:
        candidate_outputs: Outputs from candidate kernel on 2+ different inputs.
        reference_outputs: Outputs from reference model on same inputs.
        inputs_list: The input tensors used (2+ sets).
        runtime_ms: Median kernel runtime in milliseconds.

    Returns:
        (passed, reason) where reason explains the first failure, or "all checks passed".
    """
    # 1. Not a no-op
    passed, reason = check_not_noop(runtime_ms)
    if not passed:
        return False, reason

    # 2. Output shapes match reference
    if reference_outputs:
        passed, reason = check_shapes_match(candidate_outputs[0], reference_outputs[0])
        if not passed:
            return False, reason

    # 3. Output not constant (needs 2+ outputs)
    if len(candidate_outputs) >= 2:
        passed, reason = check_output_not_constant(
            candidate_outputs[0], candidate_outputs[1]
        )
        if not passed:
            return False, reason

    # 4. Not passthrough
    if inputs_list:
        passed, reason = check_not_passthrough(candidate_outputs[0], inputs_list[0])
        if not passed:
            return False, reason

    return True, "All anti-hack checks passed"
