FROM  pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /root

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

ENV PYTHONPATH "${PYTHONPATH}:/root/pixelcnn"
ENV PYTHONPYCACHEPREFIX=/tmp/cpython/
