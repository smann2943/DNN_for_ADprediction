# Use an older Ubuntu base image that is compatible with the older build tools
ARG TF_VERSION=v1.4.1

# Builder stage: compile TensorFlow and produce a wheel
FROM ubuntu:16.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    git \
    build-essential \
    pkg-config \
    zip \
    zlib1g-dev \
    software-properties-common \
    python3-dev \
    python3-pip \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.5 from deadsnakes (kept for compatibility with older TF)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1

ENV PATH="/usr/local/bin:$PATH"

# Install Bazel 0.5.4 required for TF 1.4.1
WORKDIR /opt
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh && \
    mkdir /tensorflow
