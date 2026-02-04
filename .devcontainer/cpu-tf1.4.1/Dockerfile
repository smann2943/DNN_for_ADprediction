FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive
ENV R_MAJOR_VERSION=3
ENV R_FULL_VERSION=${R_MAJOR_VERSION}.5.2
ENV PYTHON_VERSION=3.5
ENV PYTHON_FULL_VERSION=${PYTHON_VERSION}.10

# install ssl and gcc to support many pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    wget \
    curl \
    libcurl4-openssl-dev \
    libpcre2-dev \
    r-base-dev \
    gfortran \
    unzip \
    autoconf \
    automake \
    libtool \
    build-essential \
    openssh-client \
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
    libpcre3 libpcre3-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/* 


WORKDIR /tmp

# install R from source
RUN curl -O https://cran.rstudio.com/src/base/R-${R_MAJOR_VERSION}/R-${R_FULL_VERSION}.tar.gz && \
    tar -xzvf R-${R_FULL_VERSION}.tar.gz && \
    cd R-${R_FULL_VERSION} && ./configure   --prefix=/opt/R/${R_FULL_VERSION}   --enable-R-shlib   --enable-memory-profiling   --with-blas   --with-lapack && \
    make -j$(nproc) && make install && \
    ln -s /opt/R/${R_FULL_VERSION}/bin/R /usr/local/bin/R && \
    ln -s /opt/R/${R_FULL_VERSION}/bin/Rscript /usr/local/bin/Rscript


# download and untar python ${PYTHON_VERSION} into the tmp directory
RUN wget "https://www.python.org/ftp/python/${PYTHON_FULL_VERSION}/Python-${PYTHON_FULL_VERSION}.tgz" && \
    tar xzf "Python-${PYTHON_FULL_VERSION}.tgz"

# Install Python ${PYTHON_VERSION} from source (kept for compatibility with older TF)
RUN cd "Python-${PYTHON_FULL_VERSION}" && \
    ./configure --enable-optimizations --enable-shared --prefix="/opt/python${PYTHON_VERSION}" LDFLAGS="-Wl,-rpath=/opt/python${PYTHON_VERSION}/lib" && \
    make -j$(nproc) && \
    make altinstall

ENV PATH="/usr/local/bin:$PATH"

# Set python${PYTHON_VERSION} as default python
RUN update-alternatives --install /usr/bin/python python "/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" 1 && \
    update-alternatives --install /usr/bin/pip pip "/opt/python${PYTHON_VERSION}/bin/pip${PYTHON_VERSION}" 1 || true && \
    rm -rf "/tmp/Python-${PYTHON_FULL_VERSION}"

# Install TensorFlow 1.4.1 and other common packages
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
    ./bazel-0.5.4-installer-linux-x86_64.sh && \
    rm bazel-0.5.4-installer-linux-x86_64.sh  

WORKDIR /tmp/tensorflow
# Clone TensorFlow and check out the requested tag
RUN git clone --branch v1.4.1 --depth 1 https://github.com/tensorflow/tensorflow.git .

# Install Python packages that TF 1.4.1 absolutely needs at configure time
RUN "/opt/python${PYTHON_VERSION}/bin/pip${PYTHON_VERSION}" install --no-cache-dir \
    numpy==1.13.3 \
    wheel \
    mock \
    future && \
    # Critical: pre-warm numpy so the .so files are compiled and LD_LIBRARY_PATH works
    "/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" -c "import numpy; numpy.show_config()" && \
    echo "NumPy include dir: $(/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION} -c 'import numpy; print(numpy.get_include())')"

# Now run the TF configure script – force the paths so the ancient script can't get confused
RUN PYTHON_BIN_PATH="/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" \
    PYTHON_LIB_PATH="/opt/python${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/site-packages" \
    TF_NEED_CUDA=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_MKL=0 \
    TF_NEED_GCP=0 \
    TF_NEED_HDFS=0 \ 
    yes "" | ./configure

# ←←←← ADD THESE TWO LINES RIGHT HERE ←←←←
# This is the critical fix for TF 1.4.x
RUN echo 'import %workspace%/tools/bazel.rc' > .bazelrc

ENV BAZEL_BUILD_OPTS="--local_resources 3072,1.5,1.0 --jobs=1 --ram_utilization_factor=70"

# Build without --config=opt (this is the key)
RUN export PYTHON_BIN_PATH="/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" && \
    export PYTHON_LIB_PATH="/opt/python${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/site-packages" && \
    bazel build $BAZEL_BUILD_OPTS \
    --python_path="/opt/python${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" \
    --config=opt --config=noaws --config=nogcp --config=nohadoop \
    --copt=-O2 --copt=-march=native \
    --define=boosted_trees_proto=0 \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package \
    && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# install tensorflow and clean stuff up
WORKDIR /src
RUN python -m venv .venv && \
    python -m pip install --upgrade pip && \
    python -m pip install --upgrade pip && \
    python -m pip install /tmp/tensorflow_pkg/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl 

# add alternative gcc version for vscode compatibility
RUN apt-get update && apt-get install -y gnupg2 && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y gcc-9 g++-9 && \
    rm -rf /var/lib/apt/lists/*

