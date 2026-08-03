"""
Modal serverless functions for CUDA kernel evaluation.

Thin wrappers around eval_service.eval_core — all GPU logic lives there.
This file adds Modal-specific infrastructure (image, volumes, decorators).

For CoreWeave/Northflank deployment, see eval_service/app.py instead.
"""
import modal
import os
from typing import Any

from openenv_env.anti_hack import extract_cu_flags, scan_forbidden_symbols

# Forward target-hardware knobs from the deployer's shell into the eval image.
# Without this, `modal deploy` containers start clean and silently fall back
# to the sm_80/A100 defaults below — deploying with KERNELFORGE_CUDA_ARCH=sm_90a
# would still yield containers compiling sm_80. Same hazard modal_train.py
# documents for KERNELFORGE_*; image env is content-addressed, so distinct
# value combos cache distinct image revisions.
_KERNELFORGE_ENV_PASSTHROUGH = {
    k: v
    for k, v in os.environ.items()
    if k in ("KERNELFORGE_CUDA_ARCH", "KERNELFORGE_TARGET_GPU", "KERNELFORGE_TARGET_ARCH")
}

# CUDA 12.4 Docker image with Ampere/Hopper support
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "cupy-cuda12x>=14.0",
        "networkx>=3.0",
        "numpy>=1.26",
        "scipy>=1.12",
        "torch>=2.4",
        "ninja>=1.11",
        "pydantic>=2.0",
        "openenv-core[core]>=0.2.1",
    )
    .env(_KERNELFORGE_ENV_PASSTHROUGH)
    # Python packages → add_local_python_source (auto-added to PYTHONPATH at /root)
    .add_local_python_source("verification", "openenv_env", "eval_service")
    # Non-Python CUDA sources → add_local_dir
    .add_local_dir("kernels", remote_path="/root/kernels")
)

TARGET_GPU = os.getenv("KERNELFORGE_MODAL_GPU", os.getenv("KERNELFORGE_TARGET_GPU", "A100"))
TARGET_CUDA_ARCH = os.getenv("KERNELFORGE_CUDA_ARCH", os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80"))
APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")

app = modal.App(APP_NAME)
kernel_cache = modal.Volume.from_name("kernelforge-cache", create_if_missing=True)


@app.function(
    gpu=TARGET_GPU,
    image=cuda_image,
    timeout=120,
    volumes={"/cache": kernel_cache},
    include_source=True,
)
def evaluate_kernel(payload: dict) -> dict:
    """Compile, verify (PAC-reasoning), and benchmark CUDA kernel on target GPU."""
    from eval_service.eval_core import evaluate_kernel_impl
    return evaluate_kernel_impl(payload)


@app.function(gpu=TARGET_GPU, image=cuda_image, timeout=300, include_source=True)
def profile_baselines() -> dict:
    """Profile baseline kernels on the target GPU."""
    from eval_service.eval_core import profile_baselines_impl
    return profile_baselines_impl()


@app.function(gpu=TARGET_GPU, image=cuda_image, timeout=60, include_source=True)
def test_gpu_features() -> dict:
    """Test target GPU feature availability.

    Named to match the eval_backend dispatch table and the HTTP service route
    (both use "test_gpu_features") — the old "test_h100_features" deploy name
    broke Function.from_name lookups.
    """
    from eval_service.eval_core import test_gpu_features_impl
    return test_gpu_features_impl()


@app.function(
    gpu=TARGET_GPU,
    image=cuda_image,
    timeout=600,
    volumes={"/cache": kernel_cache},
    include_source=True,
)
def evaluate_kernels_batch(payloads: list[dict]) -> list[dict]:
    """Evaluate multiple kernels in a single Modal call."""
    from eval_service.eval_core import evaluate_kernels_batch_impl
    return evaluate_kernels_batch_impl(payloads)


@app.function(
    gpu=TARGET_GPU,
    image=cuda_image,
    timeout=120,
    volumes={"/cache": kernel_cache},
    include_source=True,
)
def evaluate_ops6k_kernel(payload: dict) -> dict:
    """Evaluate a CUDA kernel against a PyTorch reference from CUDA-Agent-Ops-6K."""
    from eval_service.eval_core import evaluate_ops6k_kernel_impl
    return evaluate_ops6k_kernel_impl(payload)


if __name__ == "__main__":
    # Test Modal functions locally — Function objects must be called via
    # .remote() (direct __call__ is a TypeError under modal >= 1.x).
    with app.run():
        print("Testing GPU features...")
        result = test_gpu_features.remote()
        print(f"GPU features: {result}")

        print("\nTesting baseline profiling...")
        baselines = profile_baselines.remote()
        print(f"Baselines: {baselines}")
