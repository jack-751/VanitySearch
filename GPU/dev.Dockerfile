# CUDA Development Environment
# Based on the same base image used in ccap-13.1.Dockerfile
#
# Usage (build):
#   docker build -f dev.Dockerfile --build-arg CUDA=13.1.1-cudnn-devel-ubuntu24.04 -t vanitysearch:dev .
#
# Usage (run interactive shell):
#   docker run --rm -it --gpus all -v ${PWD}:/workspace vanitysearch:dev
#
# Or use docker-compose:
#   docker compose -f docker-compose.dev.yml up -d
#   docker compose -f docker-compose.dev.yml exec dev bash

ARG CUDA=13.1.1-cudnn-devel-ubuntu24.04

FROM nvidia/cuda:${CUDA}

# Install development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    g++-13 \
    make \
    gdb \
    cuda-gdb-13-1 \
    vim \
    nano \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    libhiredis-dev \
    libmongoc-dev \
    libbson-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up compiler symlinks so 'g++' resolves to g++-13
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Default working directory is the mounted source
WORKDIR /workspace

CMD ["/bin/bash"]
