FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive

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

# download and untar python 3.5 into the tmp directory
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz && \
    tar xzf Python-3.5.10.tgz

# Install Python 3.5 from source (kept for compatibility with older TF)
RUN cd Python-3.5.10 && \
    ./configure --enable-optimizations --enable-shared --prefix=/opt/python3.6 LDFLAGS="-Wl,-rpath=/opt/python3.5/lib" && \
    make -j$(nproc) && \
    make altinstall

ENV PATH="/usr/local/bin:$PATH"

# Set python3.5 as default python
RUN update-alternatives --install /usr/bin/python python /opt/python3.5/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/pip pip /opt/python3.5/bin/pip3.5 1 || true && \
    rm -rf /tmp/Python-3.5.10

# Install TensorFlow 1.4.1 and other common packages
WORKDIR /opt
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh && \
    mkdir /tmp

# Clone TensorFlow and check out the requested tag
WORKDIR /tmp
RUN git clone --branch v1.4.1 --depth 1 https://github.com/tensorflow/tensorflow.git .

# Install Python packages that TF 1.4.1 absolutely needs at configure time
RUN /opt/python3.5/bin/pip3.5 install --no-cache-dir \
    numpy==1.16.3 \
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