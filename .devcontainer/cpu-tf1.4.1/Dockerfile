# Use an older Ubuntu base image that is compatible with the older build tools
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
    apt-get update && apt list | grep python3 && \
    apt-get install -y --no-install-recommends python3.5 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1 && \
    echo 'current python version' && \
    python --version && pip --version

ENV PATH="/usr/local/bin:$PATH"

# Install Bazel 0.5.4 required for TF 1.4.1
WORKDIR /opt
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh && \
    mkdir /tensorflow

# Clone TensorFlow and check out the requested tag
WORKDIR /tensorflow
RUN git --version && \    
    git clone --branch v1.4.1 --depth 1 https://github.com/tensorflow/tensorflow.git .

# Install Python build deps and run configure
RUN pip3 install --no-cache-dir numpy six wheel && \
    echo "\n\n\n\n\n\n\n\n" | ./configure

# Build the pip package
RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Final stage: small runtime image with the wheel installed
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies and python/pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.5 python3-pip ca-certificates libstdc++6 && \
    rm -rf /var/lib/apt/lists/* && \    
    update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1 && \
    echo 'current python version' && \
    python --version && pip --version

# Copy the built wheel from the builder stage into the final image
COPY --from=builder /tmp/tensorflow_pkg/tensorflow-1.4.1-*.whl /tmp/

# Install the wheel in the final layer (this produces a separate layer containing the installed wheel)
RUN pip3 install /tmp/tensorflow-1.4.1-*.whl && rm -f /tmp/tensorflow-1.4.1-*.whl
