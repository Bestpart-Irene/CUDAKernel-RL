#!/usr/bin/env python
"""Deep hack scan — standalone reward-hack audit for candidate CUDA kernels.

EXP-018c criterion-3 deliverable (shipped under EXP-019, see
research/experiments/EXP-019-anti-hack-integrity.md). Runs, per input file:

  1. the source-level forbidden-pattern scan (openenv_env.anti_hack.
     scan_source_forbidden), including the EXP-019 macro-indirect detector
     (token pasting, computed includes, object-macro aliasing, line splicing,
     UCN identifiers);
  2. an API-namespace grep (torch:: / at:: / c10:: / thrust::) with line
     numbers, on both the raw and macro-expanded source;
  3. an `nm -D` forbidden-symbol scan for .so inputs (fails closed when nm
     or the binary is unusable).

All patterns are imported from openenv_env.anti_hack — nothing is duplicated.

Usage:
    python scripts/deep_hack_scan.py kernel.cu [more.cu ...] [built.so] [--json-out FILE]

Output: a JSON object mapping each file path to {"clean": bool,
"reasons": [...]}. Exit status: 0 if every file is clean, 1 if any file is
dirty or unreadable, 2 on usage errors.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running as a plain script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openenv_env.anti_hack import (  # noqa: E402
    FORBIDDEN_API_NAMESPACES,
    _expand_object_macros,
    _normalize_source,
    scan_forbidden_symbols,
    scan_source_forbidden,
)

_SO_SUFFIXES = {".so", ".dylib"}


def _api_namespace_grep(source: str) -> list[str]:
    """Grep torch::/at::/c10::/thrust:: with line numbers, raw + macro-expanded."""
    reasons: list[str] = []
    compiled = [(pat, re.compile(pat)) for pat in FORBIDDEN_API_NAMESPACES]

    raw_hits: set[str] = set()
    for lineno, line in enumerate(source.splitlines(), start=1):
        for pat, rx in compiled:
            m = rx.search(line)
            if m:
                raw_hits.add(pat)
                reasons.append(
                    f"api-grep: {pat!r} at line {lineno}: {line.strip()[:120]!r}"
                )

    # Macro-indirect pass: expand object-like #defines, then re-grep. Only
    # report patterns that did NOT already hit raw (avoids duplicates).
    expanded = _expand_object_macros(_normalize_source(source))
    if expanded is None:
        reasons.append(
            "api-grep: macro expansion exceeded the size cap — suspicious "
            "macro definitions (failing closed)"
        )
    else:
        for pat, rx in compiled:
            if pat in raw_hits:
                continue
            m = rx.search(expanded)
            if m:
                reasons.append(
                    f"api-grep (macro-indirect): {pat!r} matched "
                    f"{m.group(0)!r} after macro expansion"
                )
    return reasons


def _scan_source_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"read error (failing closed): {exc}"]

    reasons: list[str] = []
    forbidden = scan_source_forbidden(source)
    if forbidden:
        reasons.append(f"source-scan: {forbidden}")
    reasons.extend(_api_namespace_grep(source))
    return reasons


def _scan_shared_object(path: Path) -> list[str]:
    if not path.exists():
        return [f"read error (failing closed): no such file: {path}"]
    forbidden = scan_forbidden_symbols(path)
    return [f"nm-scan: {forbidden}"] if forbidden else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deep_hack_scan",
        description=(
            "Audit candidate CUDA kernel sources (and optionally a built .so) "
            "for reward-hack channels."
        ),
    )
    parser.add_argument(
        "files",
        nargs="+",
        help=".cu source files to scan; a .so among them gets the nm scan",
    )
    parser.add_argument(
        "--json-out",
        metavar="FILE",
        default=None,
        help="also write the JSON verdict to FILE",
    )
    args = parser.parse_args(argv)

    verdicts: dict[str, dict[str, object]] = {}
    for raw in args.files:
        path = Path(raw)
        if path.suffix.lower() in _SO_SUFFIXES:
            reasons = _scan_shared_object(path)
        else:
            reasons = _scan_source_file(path)
        verdicts[str(path)] = {"clean": not reasons, "reasons": reasons}

    payload = json.dumps(verdicts, indent=2)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")

    return 0 if all(v["clean"] for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
