# 2026-08-10 holdout / label-migration handoff — diffs needed in files owned by other workers

Companion to the 2026-08-10 dataset-governance migration (audit findings 5+6).
Done on this side already: two dead ops6k rows relabeled `unsupported` in
`datasets/combined_kernelforge.jsonl` (`datasets/relabel_unsupported_rows.py`);
`scripts/task_pool_hash.py` (supported pool = 18: 14 ops6k + 4 wcc,
`task_pool_hash=7c9eca9b3c2d`); deterministic holdout split
`datasets/holdout_eval.jsonl` (10 rows) + `datasets/holdout_manifest.json`
(holdout hash `b1a2fe836b71`, train-side hash `de2cc67175ad`);
`evaluation/eval_model.py:_load_eval_tasks` now prefers the holdout file
(that file is outside the training/openenv/eval_core surfaces).

Three diffs below are REQUIRED in files owned by the training/openenv workers.
Without diff 1 the relabel does not hold on the `filter_supported_tasks` path;
without diffs 2–3 training still samples the held-out eval tasks.

---

## Diff 1 (REQUIRED) — training/task_support.py: explicit `unsupported` labels must be sticky

`normalize_task_row` → `infer_evaluation_backend` recomputes support from
`task_code`, so the two rows relabeled `unsupported` on 2026-08-10 (3+ tensor
inputs, no inferable extern-C signature; ops
`['F.softmax','F.scaled_dot_product_attention']` and
`['torch.stack','torch.masked_select']`) get flipped back to "ops6k" by
`filter_supported_tasks` (used by `training/dataset_loader.py` and
`evaluation/eval_model.py`'s fallback). Only the unsupported direction should
be sticky — explicit "ops6k"/"wcc" labels must still be re-verified.

```diff
--- a/training/task_support.py
+++ b/training/task_support.py
@@ def infer_evaluation_backend(row: dict[str, Any]) -> str:
 def infer_evaluation_backend(row: dict[str, Any]) -> str:
     """Infer which evaluator can score a row live."""
+    # Owner decision 2026-08-10: explicit dataset-level unsupported labels are
+    # sticky. Curated-out rows (e.g. 3+-tensor ops6k rows with no extern-C
+    # signature class — E3 deferred, see EXP-018d) must not be re-included by
+    # recomputation. Only the unsupported direction is sticky; claimed
+    # "ops6k"/"wcc" labels are still re-verified below.
+    if row.get("supports_evaluation") is False or row.get("evaluation_backend") == "unsupported":
+        return "unsupported"
     task_code = str(row.get("task_code") or "").strip()
     if task_code:
         return "ops6k" if supports_ops6k_live_eval(task_code) else "unsupported"
```

Recommended follow-up (same file, separate decision — closes the rebuild
poison AND audit finding 7's trainer/evaluator divergence at the root): make
ops6k support require an inferable signature class, so a dataset rebuild can
never resurrect signature-less rows:

```diff
--- a/training/task_support.py
+++ b/training/task_support.py
@@ def infer_evaluation_backend(row: dict[str, Any]) -> str:
     task_code = str(row.get("task_code") or "").strip()
     if task_code:
-        return "ops6k" if supports_ops6k_live_eval(task_code) else "unsupported"
+        if not supports_ops6k_live_eval(task_code):
+            return "unsupported"
+        # The evaluator only has E1/E2 extern-C signatures; a row it cannot
+        # infer a signature for is permanently dead (EXP-018d open item).
+        return "ops6k" if infer_signature_class(task_code) is not None else "unsupported"
```

(Note: `infer_signature_class` execs `get_inputs()`; if that cost matters on
hot paths, gate it behind a small cache keyed on `id(task_code)`/hash.)

## Diff 2 (REQUIRED) — training/dataset_loader.py: exclude the holdout split from training rows

Training must never see the held-out prompts. Prompt is the join key every
consumer already uses (`tasks/build_task_pool.py` comment). Apply after
`filter_supported_tasks`:

```diff
--- a/training/dataset_loader.py
+++ b/training/dataset_loader.py
@@
 DEFAULT_COMBINED_PATH = ROOT / "datasets" / "combined_kernelforge.jsonl"
 DEFAULT_DG_SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"
+DEFAULT_HOLDOUT_PATH = ROOT / "datasets" / "holdout_eval.jsonl"
+
+
+def _holdout_prompts(path: Path = DEFAULT_HOLDOUT_PATH) -> set[str]:
+    """Prompts of the held-out eval split (scripts/build_holdout_split.py).
+
+    Training must never include these rows (AGENTS.md: never train on the
+    eval split). Empty set when no holdout file exists.
+    """
+    if not path.exists():
+        return set()
+    prompts: set[str] = set()
+    with path.open() as f:
+        for line in f:
+            raw = line.strip()
+            if raw:
+                prompts.add(str(json.loads(raw).get("prompt", "")).strip())
+    prompts.discard("")
+    return prompts
@@ def load_training_dataset(
     supported_rows = filter_supported_tasks(rows)
+    holdout_prompts = _holdout_prompts()
+    if holdout_prompts:
+        before = len(supported_rows)
+        supported_rows = [r for r in supported_rows if r["prompt"] not in holdout_prompts]
+        excluded = before - len(supported_rows)
+        print(
+            f"Holdout exclusion: removed {excluded} eval-split rows from the "
+            f"training pool ({len(supported_rows)} remain; see "
+            "datasets/holdout_manifest.json)."
+        )
```

Post-exclusion training pool measured today: 8 rows (6 ops6k + 2 wcc) — the
10-row holdout minimum bites hard on an 18-row universe. `stage1`'s
`len(stage1_rows) < 16` fallback keeps it functional, but flag the 8-row pool
in the next campaign review.

## Diff 3 (REQUIRED) — openenv_env/task_pool.py: same exclusion in the TaskPool fallback

`TaskPool.load()`'s combined-dataset fallback feeds training episodes
(`KernelForgeEnv.reset`), so it must also skip holdout rows. (Its explicit
`evaluation_backend in {"ops6k","wcc"}` filter already respects the 2026-08-10
relabel — no change needed for that.)

```diff
--- a/openenv_env/task_pool.py
+++ b/openenv_env/task_pool.py
@@
 _PROJECT_ROOT = Path(__file__).resolve().parents[1]
 _DEFAULT_POOL_PATH = _PROJECT_ROOT / "tasks" / "pool_v0.jsonl"
+_HOLDOUT_PATH = _PROJECT_ROOT / "datasets" / "holdout_eval.jsonl"
+
+
+def _holdout_prompts() -> set[str]:
+    """Prompts of the held-out eval split — never sampled for episodes."""
+    if not _HOLDOUT_PATH.exists():
+        return set()
+    prompts: set[str] = set()
+    with open(_HOLDOUT_PATH, encoding="utf-8") as f:
+        for line in f:
+            line = line.strip()
+            if line:
+                prompts.add(str(json.loads(line).get("prompt", "")).strip())
+    prompts.discard("")
+    return prompts
@@ class TaskPool: def load(...)
             # Fallback: load from combined dataset, filter to supported
             combined = _PROJECT_ROOT / "datasets" / "combined_kernelforge.jsonl"
             if combined.exists():
+                holdout = _holdout_prompts()
                 with open(combined, encoding="utf-8") as f:
                     for line in f:
                         line = line.strip()
                         if not line:
                             continue
                         row = json.loads(line)
-                        if row.get("evaluation_backend") in {"ops6k", "wcc"}:
+                        if (
+                            row.get("evaluation_backend") in {"ops6k", "wcc"}
+                            and str(row.get("prompt", "")).strip() not in holdout
+                        ):
                             tasks.append(row)
```

Caveat for the same owner: if `tasks/pool_v0.jsonl` is ever generated
(`tasks/build_task_pool.py` against HF Ops-6K), it takes priority over the
fallback and is NOT holdout-filtered — its rows are drawn from the full 6K
set, so any overlap with the 8 held-out ops6k tasks must be excluded by
prompt there too before that path is used for training.

---

Verification once diffs land: `training.dataset_loader.load_training_dataset("stage3")`
must return rows whose prompt set is disjoint from `datasets/holdout_eval.jsonl`,
and `filter_supported_tasks` over the combined dataset must count 18 supported
rows (14 ops6k + 4 wcc), not 20.
