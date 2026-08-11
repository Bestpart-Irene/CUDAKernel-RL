# EXP-019 — Anti-hack integrity restoration bundle

**Status**: implemented locally, NOT committed, NOT a managed run.
**Authorization**: 2026-08-10 tri-agent audit + owner approval.
**Source audit**: `research/audits/2026-08-10-reviewer-governance-integrity.md`
(finding 2 channel list + "Residual measurement risk" section).

## This is NOT a reward-shape change

This bundle is an **integrity restoration**: four bug fixes that make the
*already-documented* v3-symbol-shaped semantics actually hold. No tier value,
no threshold in the {-1, -0.5, 0, 1, 2, 3} ladder, and no speedup cutoff
(1.05) changes. `KERNELFORGE_REWARD_VERSION` stays `v3-symbol-shaped`.
`research/live/master.json` is currently `hash=null` (no promoted master), so
there is **no comparability fork**: no prior promoted row was produced under
the buggy behaviors in a way this bundle would invalidate. Per the
kernelforge-reward-design skill, the three integrity probes are implemented as
CPU-runnable tests (`tests/test_reward_integrity_probes.py`) and pass.

Files touched (ownership boundary respected):

- `openenv_env/reward.py` (fix a)
- `openenv_env/anti_hack.py` (fixes b, c, d)
- `tests/test_reward_integrity_probes.py` (new — probes + regressions)
- `scripts/deep_hack_scan.py` (new — EXP-018c criterion-3 deliverable)
- `tests/test_anti_hack.py` (one pre-existing test tightened; see fix c)

NOT touched: `eval_service/eval_core.py`, `evaluation/verifier.py`,
`training/*`, `datasets/*`.

---

## Fix a — NaN/inf speedup guard in `compute_reward`

### OLD behavior (verbatim, `openenv_env/reward.py`)

```python
def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
    symbol_loaded: bool = True,
) -> float:
    ...docstring elided (unchanged)...
    # EXP-009 diagnostic: confirm reward chain is reached and which version is active.
    print(
        f"[COMPUTE_REWARD] compiled={compiled} correct={correct} "
        f"symbol_loaded={symbol_loaded} "
        f"sv_eager={speedup_vs_eager} sv_compile={speedup_vs_compile} "
        f"version={_REWARD_VERSION}",
        flush=True,
    )
    if not compiled:
        return -1.0
    if not correct:
        if _REWARD_VERSION == "v1-discrete-milestone":
            # Legacy binary: lumps compiled_but_wrong with compile_failed.
            return -1.0
        if _REWARD_VERSION == "v3-symbol-shaped" and not symbol_loaded:
            # EXP-010: model compiled something but the canonical contract
            # symbol is missing (e.g. forgot `extern "C"` on wcc_kernel).
            # Worse than "numerically wrong" because there's literally no
            # callable kernel; better than compile-fail because syntax/types
            # are sound. The intermediate bucket gives GRPO gradient toward
            # fixing the contract.
            return -0.5
        # v2-shaped (and v3 when symbol DID load): compile-pass + symbol
        # is a partial signal, only the numerics are wrong.
        # Kevin (arXiv 2507.11948) shows this unlocks small-model GRPO
        # cold-start; EXP-004 v3 confirmed on this stack.
        return 0.0

    # Discrete milestones (highest matching tier wins)
    if speedup_vs_compile > 1.05:
        return 3.0
    if speedup_vs_eager > 1.05:
        return 2.0
    return 1.0
```

### Failure mode

`validate_eval_result` clamps NaN/inf, but `compute_reward` is also reached
via `training/task_support.compute_task_reward`, which passes
`float(normalized.get("speedup_vs_dg") or 0.0)` — `float('inf')` is truthy and
finite-checked nowhere on that path. In Python, `float('inf') > 1.05` is
`True`, so a corrupted/hacked timing path that yields `speedup_vs_dg = inf`
(e.g. a zero/denormal baseline division upstream of the guarded division, or a
backend returning `Infinity` in JSON) maps straight to **reward 3.0** — the
maximum reward for a measurement that never happened. `NaN` compares `False`
on both tiers and silently landed on 1.0 — the right value by accident, with
no record that the measurement was garbage.

### NEW behavior

Both speedups are sanitized at the top of `compute_reward`: any value that is
not a real finite number (NaN, +inf, -inf, non-numeric) is replaced with
`0.0` — i.e. *"no speedup was measured"* — before the tier ladder runs. A
correct kernel with a non-finite speedup therefore earns exactly the
correctness-tier reward **1.0**, never 2.0/3.0. A one-line
`[COMPUTE_REWARD] non-finite speedup clamped ...` diagnostic is printed when
the guard fires. `validate_eval_result` is unchanged (it already clamped);
the guard now holds at the last line of defense regardless of entry path.

---

## Fix b — `check_not_noop` dead threshold (units made explicit)

### OLD behavior (verbatim, `openenv_env/anti_hack.py`)

```python
def check_not_noop(runtime_ms: float) -> tuple[bool, str]:
    """Flag suspiciously fast kernels that likely skip computation.

    Any real CUDA kernel launch takes at least ~2μs due to launch overhead.
    If median runtime is below 0.001ms (1μs), the kernel is almost certainly
    a no-op.

    Returns (passed, reason).
    """
    if runtime_ms < 0.001:
        return False, f"Runtime {runtime_ms:.4f}ms is below 1μs — likely a no-op kernel"
    return True, f"Runtime {runtime_ms:.3f}ms is plausible"
```

### Failure mode (the units bug)

The runtime fed to this check is `torch.cuda.Event.elapsed_time()` /
`cupy.cuda.get_elapsed_time()` output — **milliseconds** with ~0.5μs timer
resolution (`eval_service/eval_core.py:931-935,955`). The docstring's own
premise is that a kernel *launch alone* costs ~2μs, yet the threshold was
written as `0.001` ms = **1μs — below the physical floor the docstring
cites**. CUDA-event timing of *any* code path — even an empty launch, even
back-to-back event records with nothing between them — never reads
meaningfully below ~1-2μs, so `runtime_ms < 0.001` is unsatisfiable in
practice: the check is dead code (confirmed by audit finding 2, "beating
check_not_noop's 1μs floor"). The 2μs-overhead → 0.001ms translation dropped
a factor of 2 and placed the cutoff below, not at, the launch-overhead floor.
Additionally, `float('nan') < 0.001` is `False`, so a NaN runtime **passed**
the no-op check.

### NEW behavior

The conversion is explicit: module constants `_US_PER_MS = 1000.0` and
`_MIN_PLAUSIBLE_KERNEL_TIME_US = 2.0` (the launch-overhead floor the original
docstring already asserted). `check_not_noop` converts `runtime_ms` to μs and
fails anything strictly below 2μs, plus fails closed on non-numeric,
non-finite, zero, and negative runtimes (a reading below or at zero, or NaN,
means timing was subverted or broken — hack-suspect either way). The dead
zone [1μs, 2μs) — physically impossible for real work, previously passing —
is now caught. Threshold rationale: legitimate ops6k eval workloads measure
well above 2μs (typically ≥10μs median); the floor sits below any real
workload but at the physical minimum of a launch.

**Residual (honest)**: a dummy 1-element kernel measuring ~3-4μs still beats
*any* physically-sound time floor. Closing the call-count timing-gaming
channel (audit finding 2, "timing gaming via call-count state") requires
re-verifying one timed benchmark output inside
`eval_service/eval_core.py` — a forbidden file for this bundle. Recorded as
open; needs its own authorized evaluator experiment.

---

## Fix c — `scan_forbidden_symbols` fails closed on nm failure

### OLD behavior (verbatim, `openenv_env/anti_hack.py`)

```python
def scan_forbidden_symbols(so_path: str | Path) -> str | None:
    """Return a failure reason if forbidden dynamic symbols are detected."""
    so_path = str(so_path)
    try:
        proc = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        # Do not hard-fail if nm is unavailable; caller can decide fallback behavior.
        return None

    symbols = proc.stdout
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in symbols:
            return f"Forbidden symbol detected: {forbidden}"
    return None
```

### Failure mode

Two silent-pass holes:

1. `except Exception: return None` — `nm` not installed, not on PATH, or a
   5s timeout ⇒ the scan **passes** every binary. The comment says "caller
   can decide fallback behavior", but no caller can: `None` is
   indistinguishable from a genuinely clean scan at every call site
   (`eval_service/eval_core.py:356,749`, `evaluation/verifier.py:36`).
2. `proc.returncode != 0` was never inspected — `nm` on a **missing or
   unreadable .so** exits nonzero with empty stdout, and empty stdout
   trivially contains no forbidden substring ⇒ **PASS**. A candidate whose
   binary vanished (or an image without working nm) sailed through the
   post-link anti-hack layer.

This is the audit's "silent environment fault degrades hack detection to
zero without signal" class.

### NEW behavior

Any nm failure — exception (missing binary path is pre-checked too),
`FileNotFoundError` for nm itself, timeout, or nonzero exit — returns a
failure string prefixed `hack_suspected: symbol scan failed ...` with the
underlying error recorded (exception repr or nm exit code + stderr head).
All existing callers treat a non-`None` return as a failed verdict, so the
fail-closed semantics propagate without touching any forbidden file. A clean
scan (nm exit 0, no forbidden substring) still returns `None`.

**Operational note**: an eval image without working `nm` now rejects every
candidate (loudly, with the reason in `error`/`verifier_msg`) instead of
passing every candidate silently. That is the intended direction for a
detection layer; the image already ships binutils.

### Pre-existing test updated

`tests/test_anti_hack.py::test_scan_nonexistent_path` previously asserted
`result is None or isinstance(result, str)` — i.e. it *tolerated* the buggy
silent PASS on a nonexistent path. Tightened to require a failure string
containing `hack_suspected` (the missing-binary case must now fail closed).
No other pre-existing test asserted the old behavior of any of the four
fixes (`scripts/smoke_test.py`'s 13 reward assertions use only finite
speedups and are unaffected).

---

## Fix d — Macro-indirect include / identifier bypass of the source scan

### OLD behavior (verbatim, `openenv_env/anti_hack.py`)

```python
FORBIDDEN_SOURCE_PATTERNS = [
    r'#\s*include\s*[<"]torch/',
    r'#\s*include\s*[<"]ATen/',
    r'#\s*include\s*[<"]c10/',
    r'\btorch::',
    r'\bat::',
    r'\bc10::',
]


def scan_source_forbidden(source: str) -> str | None:
    """Return rejection reason if source matches a forbidden pattern, else None.

    Source-level pre-compile check for the unified extern "C" eval contract
    (EXP-018a). Closes the torch::-API delegation reward-hack channel that
    invalidated EXP-015'-A. Must run BEFORE nvcc to keep `nm -D` scans
    meaningful (no libtorch linkage means no false positives).
    """
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        m = re.search(pat, source)
        if m:
            return f"forbidden source pattern: {pat!r} matched {m.group(0)!r}"
    return None
```

### Failure mode

The scan ran the regexes over the **raw text only**, so any preprocessor
indirection evaded all six patterns while nvcc still saw the forbidden
construct:

- computed include: `#define TORCH_HDR <torch/extension.h>` +
  `#include TORCH_HDR` — no literal `#include <torch/` anywhere;
- object-macro aliasing: `#define NS torch` + `NS::relu(...)`;
- token pasting: `#define NS tor##ch` (or a function-like
  `#define CAT(a,b) a##b` + `CAT(tor,ch)::...`);
- line splicing: `#include <tor\` newline `ch/extension.h>` — the
  preprocessor joins backslash-newline before tokenizing, the regex does not;
- universal character names: `torch::relu` is the identifier `torch` to
  the compiler (`c` = `c`), not to a substring match.

Because the eval path has no libtorch link step, source scan evasion on the
ops6k path fails at link time in practice — but on any path where the
forbidden library IS linkable (or header-only, e.g. thrust-style), this scan
was the only gate, and it was string-literal-shallow.

### NEW behavior (detector rules D1-D5, layered on the unchanged raw scan)

`scan_source_forbidden` keeps pass 0 (raw-text scan, byte-for-byte the old
behavior, checked first) and adds:

- **D1 — reject token pasting outright**: any `##` in the comment-stripped,
  line-spliced source is rejected (`token-pasting operator '##' is not
  permitted`). Pragmatic over-block: the extern "C" eval contract gives a
  generated kernel no legitimate need for `##`, and rejecting the operator
  closes *both* object-like and function-like paste composition (the latter
  is not simulatable by a regex-level expander).
- **D2 — reject computed includes**: `#include` whose next token is not `<`
  or `"` (i.e. `#include SOME_MACRO`) is rejected. No legitimate use under
  the contract; catches `#define X "torch/extension.h"` + `#include X`
  without needing to resolve what X expands to.
- **D3 — normalization pass, then rescan the same 6 patterns**:
  line-splice (`\\\n` removed) → comments stripped to a single space (as the
  real preprocessor does) → UCN escapes `\uXXXX`/`\UXXXXXXXX` decoded to
  their characters → `##` operators collapsed (adjacent tokens joined).
  Matches found only in normalized text are reported as
  `forbidden source pattern (macro-indirect)`.
- **D4 — object-like `#define` fixed-point expansion**: object-like macro
  definitions (name not immediately followed by `(`) are substituted into
  the normalized text to a fixed point (max 8 iterations), then the 6
  patterns are rescanned. Catches `#define NS torch` / `#define A tor` +
  `#define NS A##ch` chains.
- **D5 — expansion blowup fails closed**: if expansion grows the text past
  `max(4x original, 256 KiB)`, the source is rejected as suspicious
  (self-referential/exponential macro games are not a legitimate kernel).

Ordering note (compiler-accurate): comments become a *space* (so plain
`tor/**/ch` stays two tokens — no false positive), but `##` joins across
that space (so `tor ## /**/ ch` still pastes to `torch` — caught).

A shared constant `FORBIDDEN_API_NAMESPACES`
(`torch::`, `at::`, `c10::`, `thrust::`) is added for
`scripts/deep_hack_scan.py`'s audit grep. It is deliberately **NOT** wired
into the eval-path scan: adding `thrust::` to the reward path would be a
semantics change (thrust was never in `FORBIDDEN_SOURCE_PATTERNS`), and this
bundle is restoration-only.

### Residual limits of the detector (honest)

- **Function-like macros with arguments are not expanded** (D1's blanket
  `##` ban covers the paste-based ones; a function-like macro that builds a
  forbidden name *without* `##` is not possible in standard C preprocessing,
  but argument-substitution games combined with D2-exempt includes are not
  modeled).
- **Conditional-compilation games** (`#if`/`#elif` with redefinition,
  `__has_include`) are not evaluated; a macro defined only under a
  conditional still gets expanded by D4 (over-approximation, fail-closed
  direction).
- **String/char literals are not lexed**: comment-strip can eat into a
  literal containing `/*` or `//`; UCN decoding also applies inside
  literals. Both are detection-side over-approximations (may over-reject
  pathological sources, never under-reject).
- **Trigraphs/digraphs** (`%:include` for `#include`) are not decoded
  (nvcc's default dialect post-C++17 has no trigraphs; digraph `%:` is
  listed as a known gap).
- **`-D` command-line macros** would bypass everything, but
  `extract_cu_flags`'s whitelist has no `-D` and never did.
- The neighbor channels from audit finding 2 — reference-output scraping via
  the shared CUDA context, call-count timing gaming, tolerance gaming with
  `--use_fast_math`, fixed eval seeds — live in `eval_service/eval_core.py`
  and remain **open**; out of scope (forbidden file) and unchanged here.

---

## Integrity probes (skill step 4)

`tests/test_reward_integrity_probes.py` — CPU-only, no GPU, no network:

1. **known-good** synthetic eval result (compiles, correct, speedup > both
   baselines) → reward **3.0** via both `validate_eval_result` +
   `compute_reward` and the production `compute_task_reward` dict path;
2. **known-broken** (compile fail) → reward **-1.0**;
3. **ref-copy canary**: a result flagged hack-suspected (anti-hack layer set
   `correct=False`, error `Anti-hack: ...`, inflated speedups still present
   in the dict) → reward **must not be positive** (v3 maps it to 0.0).

Plus regression tests for fixes a-d (non-finite speedup tiers, dead-zone
runtimes, nm failure modes, five macro-indirect evasion sources + one
legitimate kernel that must still pass).

## Deliverable: `scripts/deep_hack_scan.py` (EXP-018c criterion 3)

Standalone CLI; imports every pattern from `openenv_env.anti_hack` (no
duplication). Per file: (1) source forbidden-pattern scan including the
macro-indirect detector, (2) `torch::`/`at::`/`c10::`/`thrust::` API grep
with line numbers, (3) `nm -D` symbol scan for `.so` inputs. Emits a JSON
verdict per file `{"clean": bool, "reasons": [...]}`; exit 1 if any file is
dirty, 2 on usage/read errors.

```
usage: python scripts/deep_hack_scan.py kernel.cu [more.cu ...] [built.so] [--json-out FILE]
```
