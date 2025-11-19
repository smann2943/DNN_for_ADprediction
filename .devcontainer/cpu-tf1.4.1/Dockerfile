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
    ca-certificates && \
    libncurses5-dev && \
    libgdbm-dev && \
    libnss3-dev && \
    libssl-dev && \
    libreadline-dev && \
    libffi-dev && \
    libbz2-dev && \
    libsqlite3-dev && \
    liblzma-dev && \
    tk-dev && \
    uuid-dev && \
    libgdbm-compat-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.5 from source (kept for compatibility with older TF)
RUN  /usr/src && \
    wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz && \
    tar xzf Python-3.5.10.tgz && \
    cd Python-3.5.10 && \   
    ./configure --enable-optimizations --enable-shared --prefix=/opt/python3.5 LDFLAGS="-Wl,-rpath=/opt/python3.5/lib" && \
    make -j$(nproc) && \
    make altinstall    # ← important: altinstall, not install

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
RUN bazel build --python_path=/opt/python3.5/bin/python3.5 --config=opt //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Final stage: small runtime image with the wheel installed
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies and python/pip
# Copy the built wheel from the builder stage into the final image
COPY --from=builder /tmp/tensorflow_pkg/tensorflow-1.4.1-*.whl /tmp/
COPY --from=builder /opt/python3.5 /opt/python3.5

# Install the wheel in the final layer (this produces a separate layer containing the installed wheel)
RUN /opt/python3.5/bin/python3.5 -m pip install /tmp/tensorflow-1.4.1-*.whl && rm -f /tmp/tensorflow-1.4.1-*.whl && \
    update-alternatives --install /usr/bin/python python /opt/python3.5/bin/python3.5 1
