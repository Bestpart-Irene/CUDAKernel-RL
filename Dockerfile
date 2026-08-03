# KernelForge-OpenEnv Docker Image
# CUDA 12.4.1 — matches both Modal images (nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04)
# so local containers stay comparable with Modal train/eval containers.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies.
# python3.12 is NOT in Ubuntu 22.04's default apt repos — it comes from the
# deadsnakes PPA (software-properties-common provides add-apt-repository).
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Consistent python symlinks: /usr/local/bin shadows /usr/bin on PATH, so both
# `python` and `python3` resolve to 3.12 without breaking apt's /usr/bin/python3
# (which must stay Ubuntu's 3.10).
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python dependencies into the python3.12 interpreter explicitly —
# `uv pip install --system` alone would target the base image's python3.10.
# --break-system-packages guards against deadsnakes' EXTERNALLY-MANAGED marker.
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --python /usr/bin/python3.12 --break-system-packages -r /tmp/requirements.txt \
    && uv pip install --python /usr/bin/python3.12 --break-system-packages "cupy-cuda12x>=14.0"

# Create application directory
WORKDIR /app

# Copy application code (see .dockerignore for exclusions)
COPY . /app/

# Create directories for outputs and cache
RUN mkdir -p /app/outputs /app/cache /app/datasets

# Set permissions
RUN chmod +x /app/demo/streamlit_demo.py
RUN chmod +x /app/training/*.py
RUN chmod +x /app/datasets/*.py
RUN chmod +x /app/verification/*.py

# Expose ports for Streamlit demo and OpenEnv server
EXPOSE 8501
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import modal, cupy, networkx; print('OK')" || exit 1

# Default command — env-switchable: KERNELFORGE_MODE=server for OpenEnv HTTP server
CMD ["sh", "-c", "if [ \"$KERNELFORGE_MODE\" = 'server' ]; then python -m uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000; else python -m streamlit run demo/streamlit_demo.py --server.address 0.0.0.0 --server.port 8501; fi"]
