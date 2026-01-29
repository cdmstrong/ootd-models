# Example Dockerfile for OOTD service
# This Dockerfile pre-downloads all models during build time

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

    # 让 python = python3（很重要）
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升级 pip
RUN python -m pip install --upgrade pip

# Copy requirements first (for better caching)
# COPY wheels /app/wheels
RUN python -m pip install torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/cu121
# RUN pip install /app/wheels/*.whl
# RUN rm -rf /app/wheels
# COPY ./.u2net /root/.u2net
COPY requirements.txt .
# COPY diffusers /app/diffusers

RUN pip install -r requirements.txt
# RUN pip install /app/diffusers

# Install Python dependencies

# Pre-download rembg model (this will download to ~/.u2net/)
# RUN python -c "from rembg import new_session; new_session('')" || echo "Warning: rembg model download failed"

# Copy application code

# Create output directories
RUN mkdir -p app/outputs/bg_removed app/outputs
RUN pip install -U "huggingface_hub[hf_transfer]" -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN hf download Gulraiz00/u2net --local-dir ~/.u2net
RUN hf download black-forest-labs/FLUX.2-klein-4B --local-dir /app/flux2-klein
COPY . /app

CMD ["python", "handler.py"]
# Expose ports
# EXPOSE 8000 8001

# # Default command (can be overridden)
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

