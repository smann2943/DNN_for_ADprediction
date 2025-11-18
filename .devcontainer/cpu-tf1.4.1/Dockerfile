# Use an older Ubuntu base image that is compatible with the older build tools
FROM ubuntu:16.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    build-essential \
    pkg-config \
    zip \
    zlib1g-dev \
    # For Python 3.6 support
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3.6-venv && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.6 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

# Install Bazel 0.5.4 (the recommended version for TensorFlow 1.4.0/1.4.1)
# Note: Bazel installation for older versions can be tricky, using the official installer script for 0.5.4
WORKDIR /opt
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh
ENV PATH="/usr/local/bin:$PATH"

# --- TensorFlow Source and Build ---
WORKDIR /tensorflow
# Clone the repository and checkout the specific tag for 1.4.1
RUN git clone https://github.com/tensorflow/tensorflow.git . && \
    git fetch --tags && git checkout tags/1.4.1 

# Install package dependencies for building
RUN pip3 install numpy six wheel

# Configure the build (set defaults for CPU-only build)
# The default uses GCC 4.8 from Ubuntu 16.04 which is compatible.
# This script will prompt for configuration, using echo to provide default/CPU-only answers
RUN echo "\n\n\n\n\n\n\n\n" | ./configure

# Build the pip package (CPU-only)
# Using 'bazel build' and then 'bazel-bin/tensorflow/tools/pip_package/build_pip_package'
# This is the command structure for older TF versions.
# The build may take a long time and a lot of RAM.
RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# --- Final Installation ---
# Install the generated .whl file
RUN pip3 install /tmp/tensorflow_pkg/tensorflow-1.4.1-*.whl

# You can now use this image for your project.
# The installed TensorFlow is globally available in this container.
# If you need to install other dependencies from requirements.txt:
# COPY requirements.txt /app/
# RUN pip3 install -r /app/requirements.txt