# Use an older Ubuntu base image that is compatible with the older build tools
# Builder stage: compile TensorFlow and produce a wheel
FROM ubuntu:16.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install system build dependencies (include SSL and other libs required to build Python's ssl module)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    git \
    autoconf \
    automake \
    libtool \
    build-essential \
    pkg-config \
    zip \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    tk-dev \
    software-properties-common \
    ca-certificates    

# Create tmp directory
RUN  add-apt-repository universe && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    gfortran && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /tmp

WORKDIR /tmp

# download and untar python 3.5 into the tmp directory
RUN wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz && \
    tar xzf Python-3.5.10.tgz    

# Install Python 3.5 from source (kept for compatibility with older TF)
RUN cd Python-3.5.10 && \
    ./configure --enable-optimizations --enable-shared --prefix=/opt/python3.5 LDFLAGS="-Wl,-rpath=/opt/python3.5/lib" && \
    make -j$(nproc) && \
    make altinstall

ENV PATH="/usr/local/bin:$PATH"

# Set Python3.5 as default python
RUN update-alternatives --install /usr/bin/python python /opt/python3.5/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/pip pip /opt/python3.5/bin/pip3.5 1 || true

# Install Bazel 0.5.4 required for TF 1.4.1
WORKDIR /opt
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh && \
    mkdir /tensorflow

# Clone TensorFlow and check out the requested tag
WORKDIR /tensorflow
RUN git clone --branch v1.4.1 --depth 1 https://github.com/tensorflow/tensorflow.git .

# Install Python packages that TF 1.4.1 absolutely needs at configure time
RUN /opt/python3.5/bin/pip3.5 install --no-cache-dir \
    numpy==1.13.3 \
    wheel \
    mock \
    future && \
    # Critical: pre-warm numpy so the .so files are compiled and LD_LIBRARY_PATH works
    /opt/python3.5/bin/python3.5 -c "import numpy; numpy.show_config()" && \
    echo "NumPy include dir: $(/opt/python3.5/bin/python3.5 -c 'import numpy; print(numpy.get_include())')"

# Now run the TF configure script – force the paths so the ancient script can't get confused
RUN PYTHON_BIN_PATH=/opt/python3.5/bin/python3.5 \
    PYTHON_LIB_PATH=/opt/python3.5/lib/python3.5/site-packages \
    TF_NEED_CUDA=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_MKL=0 \
    TF_NEED_VER=2 \
    yes "" | ./configure

# ←←←← ADD THESE TWO LINES RIGHT HERE ←←←←
# This is the critical fix for TF 1.4.x
RUN echo 'import %workspace%/tools/bazel.rc' > .bazelrc

ENV BAZEL_BUILD_OPTS="--local_resources 3072,1.5,1.0 --jobs=2 --ram_utilization_factor=70"

# Build without --config=opt (this is the key)
RUN export PYTHON_BIN_PATH=/opt/python3.5/bin/python3.5 && \
    export PYTHON_LIB_PATH=/opt/python3.5/lib/python3.5/site-packages && \
    bazel build $BAZEL_BUILD_OPTS \
    --python_path=/opt/python3.5/bin/python3.5 \
    --config=noaws --config=nogcp --config=nohadoop \
    --copt=-O2 --copt=-march=native \
    --define=boosted_trees_proto=0 \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package \
    && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Final stage: small runtime image with the wheel installed
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /tmp/Python-3.5.10
# Install runtime dependencies and python/pip
# Copy the built wheel from the builder stage into the final image
COPY --from=builder /tmp/tensorflow_pkg/tensorflow-1.4.1-*.whl /tmp/
COPY --from=builder /tmp/Python-3.5.10 /tmp/Python-3.5.10

# install ssl and gcc to support many pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    git \
    gcc \
    autoconf \
    automake \
    libtool \
    build-essential \
    pkg-config \
    zip \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    tk-dev \
    software-properties-common \
    ca-certificates \
    r-base && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /tmp
# Install Python 3.5 from source (kept for compatibility with older TF)
RUN cd Python-3.5.10 && \
    ./configure --enable-optimizations --enable-shared --prefix=/opt/python3.5 LDFLAGS="-Wl,-rpath=/opt/python3.5/lib" && \
    make -j$(nproc) && \
    make altinstall

ENV PATH="/usr/local/bin:$PATH"

WORKDIR /
# Install the wheel in the final layer (this produces a separate layer containing the installed wheel)
RUN /opt/python3.5/bin/python3.5 -m pip install --no-deps /tmp/tensorflow-1.4.1-*.whl && rm -f /tmp/tensorflow-1.4.1-*.whl && \
    update-alternatives --install /usr/bin/python python /opt/python3.5/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/pip pip /opt/python3.5/bin/pip3.5 1 || true && \
    rm -rf /tmp/Python-3.5.10
