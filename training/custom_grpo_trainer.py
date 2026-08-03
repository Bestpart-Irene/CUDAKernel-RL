"""
TRLOO-augmented GRPOTrainer — fixes the 25% gradient shrinkage from GRPO self-inclusion bias.

Dr. Kernel (arXiv 2602.05885) proves that GRPO's advantage estimation includes
the current sample in its own baseline, causing E[gradient] = (1 - 1/N) * true_gradient.
With G=4, gradients are systematically 25% too small.

Fix: scale advantages by N/(N-1) after GRPO computes them. This is the TRLOO
(Turn-level Reinforce Leave-One-Out) correction — mathematically equivalent to
computing the baseline from the other G-1 samples.

MARS return-to-go was considered but dropped: with outcome-only rewards (no per-turn
signal), MARS degenerates to standard trajectory-level GRPO (see ALPHXIV analysis).
"""
from __future__ import annotations

import warnings

import torch
from trl import GRPOTrainer, GRPOConfig


def trloo_hook_available() -> bool:
    """True when the installed TRL's GRPOTrainer exposes `_compute_advantages`.

    TRL 0.29 computes advantages inline in grpo_trainer.py — the hook does not
    exist there, so the override below is never called and the TRLOO
    correction silently does nothing (2026-08-03 audit).
    """
    return hasattr(GRPOTrainer, "_compute_advantages")


class TRLOOGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with TRLOO advantage correction.

    Drop-in replacement: just swap GRPOTrainer → TRLOOGRPOTrainer.
    Everything else (reward_funcs, rollout_func, config) stays the same.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TRLOO N/(N-1) is the *vanilla* GRPO self-inclusion correction. DAPO
        # (TRL 0.29 default) already applies an unbiased estimator that removes
        # the same shrinkage; stacking TRLOO on top inflates advantages by an
        # extra G/(G-1). Only enable when the loss is vanilla "grpo".
        loss_type = getattr(self.args, "loss_type", "grpo")
        self._trloo_enabled = loss_type == "grpo"
        self._trloo_hook_available = trloo_hook_available()
        self._trloo_active = self._trloo_hook_available and self._trloo_enabled
        if not self._trloo_hook_available:
            msg = (
                "TRLOOGRPOTrainer: base GRPOTrainer has NO `_compute_advantages` "
                "hook in this TRL version (0.29 computes advantages inline in "
                "grpo_trainer.py) — the TRLOO N/(N-1) correction is INACTIVE and "
                "training runs with vanilla TRL advantages. Note also the "
                'internal gate only enables TRLOO for loss_type=="grpo", while '
                f'TRL 0.29 defaults to loss_type="dapo" (this run: '
                f"loss_type={loss_type!r})."
            )
            print(f"WARNING: {msg}", flush=True)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages with TRLOO N/(N-1) correction.

        TRL's GRPOTrainer computes advantages as:
            A_i = (r_i - mean(r)) / (std(r) + eps)

        This includes sample i in its own baseline, causing (1-1/N) gradient shrinkage.
        We apply the correction after the base computation, but ONLY for
        loss_type="grpo" — DAPO is already unbiased.

        Only reachable on TRL versions whose GRPOTrainer actually exposes this
        hook — TRL 0.29 does not (see trloo_hook_available()).
        """
        parent_compute = getattr(super(), "_compute_advantages", None)
        if parent_compute is None:
            raise RuntimeError(
                "TRLOOGRPOTrainer._compute_advantages was called, but the base "
                "GRPOTrainer has no such hook in this TRL version — there is no "
                "vanilla-advantage computation to correct here."
            )
        # Let parent compute vanilla GRPO advantages
        advantages = parent_compute(rewards)

        if not self._trloo_enabled:
            return advantages

        # Apply TRLOO correction per group
        # rewards shape: (batch_size * num_generations,)
        # Each group of num_generations consecutive entries shares a prompt
        G = self.args.num_generations
        if G <= 1:
            return advantages

        scale = G / (G - 1.0)

        # Scale all advantages — the N/(N-1) factor is uniform within each group
        advantages = advantages * scale

        return advantages


def create_trloo_trainer(
    model,
    tokenizer,
    reward_funcs,
    train_dataset,
    config: GRPOConfig,
    rollout_func=None,
) -> TRLOOGRPOTrainer:
    """Factory function to create a TRLOO-augmented GRPO trainer.

    Args:
        model: The model to train (with LoRA already applied).
        tokenizer: The tokenizer / processing_class.
        reward_funcs: Reward function(s) for GRPO.
        train_dataset: HF Dataset with 'prompt' column.
        config: GRPOConfig with training hyperparameters.
        rollout_func: Optional custom rollout (for multi-turn OpenEnv).

    Returns:
        TRLOOGRPOTrainer ready to .train()
    """
    kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": reward_funcs,
        "args": config,
        "train_dataset": train_dataset,
    }
    if rollout_func is not None:
        kwargs["rollout_func"] = rollout_func

    return TRLOOGRPOTrainer(**kwargs)
