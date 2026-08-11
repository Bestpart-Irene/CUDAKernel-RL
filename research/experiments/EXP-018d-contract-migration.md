# EXP-018d — ops6k prompt/contract migration to unified extern "C"

**Date:** 2026-08-03
**Type:** bugfix / dataset hygiene (no training-loop change)
**Parent:** EXP-018a/c (unified extern "C" + dlsym contract)

## Problem

The 2026-08-03 audit found the EXP-018 migration updated the evaluator but not
the three places that tell the model what to write, leaving the live 16-task
ops6k pool internally contradictory:

1. **Dataset prompts** (15/16 live ops6k rows): still demanded
   `#include <torch/extension.h>` + `PYBIND11_MODULE`, which
   `eval_core` hard-rejects at the source scan → reward pinned at −1 for a
   model that follows instructions.
2. **Contract text** (`task_support.task_interface_contract`): after 51eb7c9
   dropped `extern_c_signature` from the dataset, the fallback hardcoded E1
   while the evaluator independently infers E1/E2 from `get_inputs()` —
   two-input tasks got mis-instructed (args misbound: `out` received as `n`).
3. **Prompt builder** (`cuda_agent_integration._build_cuda_prompt`): never
   migrated, so any dataset rebuild reproduced the poison.

## Fix

- `training/task_support.py`
  - `infer_signature_class(task_code)`: mirrors
    `eval_core._infer_extern_c_signature` (exec + tensor count when torch is
    available) with a torch-free AST fallback (elements of `get_inputs()`'s
    return literal).
  - `ops6k_contract_text(sig_class)`: single source of truth for the contract
    text; `task_interface_contract` now uses explicit signature → inference →
    E1 last resort.
  - `build_generation_prompt`: skips appending the contract when the row
    prompt already embeds it (post-migration rows).
- `training/cuda_agent_integration.py`: `_build_cuda_prompt` emits the
  extern-C contract via the shared helpers (inference runs before truncation).
- `scripts/migrate_ops6k_prompts_extern_c.py`: rewrites ONLY the `prompt`
  column of live ops6k rows still carrying the legacy demands. `.bak` +
  atomic write + idempotent.

## Result

- 15/225 rows migrated; second run migrates 0 (idempotent).
- Post-state: 16 live ops6k rows, 0 demand pybind; field-level diff vs `.bak`
  is `{prompt}` only; hand-curated `vector_add_e2` untouched.
- Tests: 199 passed (8 new in `tests/test_exp018_contract.py`), smoke 8/8.

## Open items — RESOLVED (owner decision 2026-08-10)

Two live ops6k rows had 3+ tensor inputs, so neither the contract generator
nor the evaluator could produce a signature (evaluator fails with "Could not
infer extern_c_signature" — permanently dead "supported" tasks):

- `['F.softmax', 'F.scaled_dot_product_attention']`
- `['torch.stack', 'torch.masked_select']`

**Decision taken 2026-08-10: both rows relabeled `unsupported`
(`datasets/relabel_unsupported_rows.py` — atomic, idempotent: run 1 relabeled
2/225, run 2 relabeled 0/225, no `.bak` left). An E3 signature class is
deferred.** The relabel script asserts `infer_signature_class` still returns
None for both rows before touching them.

Measured post-relabel (2026-08-10, `datasets/combined_kernelforge.jsonl`,
explicit `evaluation_backend` labels): 14 ops6k + 4 wcc = 18 supported rows of
225 total, 207 unsupported — the effective live ops6k pool is now honestly
14 supported tasks. `scripts/task_pool_hash.py` reports
`task_pool_hash=7c9eca9b3c2d` over the 18-row supported union.

Caveat: `training/task_support.normalize_task_row` recomputes support from
`task_code` and would flip these two rows back to "ops6k" on the
`filter_supported_tasks` path; the sticky-label diff is routed via
`research/audits/2026-08-10-holdout-handoff.md` (training/ owned by another
worker). `openenv_env/task_pool.TaskPool.load`'s fallback filters on the
explicit dataset field, so the relabel already holds there.
