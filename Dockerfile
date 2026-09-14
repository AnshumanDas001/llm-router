# Multi-arch: the python:3.12-slim base has both amd64 and arm64 variants, so
# this builds natively on an Oracle Ampere (arm64) box and on x86 alike.
#
# 3.12 rather than the 3.14 used in local dev: the code needs nothing newer
# than 3.10 syntax, and prebuilt arm64 wheels for torch/scikit-learn lag on
# the newest Python. A missing wheel means a from-source build that takes an
# hour and usually fails on a small VM.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # sentence-transformers downloads all-MiniLM-L6-v2 on first use; keep it
    # in a known place so a volume can cache it across container restarts.
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install torch from the CPU-only index BEFORE the rest of requirements.
# The default PyPI torch on Linux is the CUDA build -- 2-3 GB of NVIDIA
# libraries that a CPU host never uses. This is the single biggest lever on
# image size.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# The DB and the model cache live on a volume mounted here (see compose).
RUN mkdir -p /app/logs /app/.cache/huggingface

EXPOSE 8000

# One worker, on purpose: the embedding classifier is ~600 MB resident and
# SQLite wants a single writer. Concurrency comes from FastAPI's threadpool
# inside the one process. --proxy-headers so client IPs survive Caddy.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
