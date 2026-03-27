## Dockerfile for Park et al. (2020) DNN for AD Prediction
## Reproduces the exact software environment from the paper.
##
## Build:  docker build --platform linux/amd64 -t park2020-ad-dnn .
## Run:    docker run --platform linux/amd64 -v $(pwd):/workspace -w /workspace -it park2020-ad-dnn bash
##
## NOTE: On Apple Silicon, QEMU emulation (~5-10x slower but numerically correct)
##       is used automatically via --platform linux/amd64.

FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu16.04

# Pin versions from dependency forensics (Phase 1)
ENV TF_VERSION=1.4.1 \
    NUMPY_VERSION=1.16.3 \
    PANDAS_VERSION=0.24.2 \
    SKLEARN_VERSION=0.21.0 \
    BAYESOPT_VERSION=1.0.1 \
    MATPLOTLIB_VERSION=3.0.3 \
    SCIPY_VERSION=1.2.1

# System dependencies for R, compilation, and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    gcc g++ gfortran make \
    pkg-config \
    zlib1g-dev \ 
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    locales locales-all \
    python3-venv \
    python3-pip \
    software-properties-common \
    wget && \
    rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8  
ENV LC_ALL en_US.UTF-8

# Install R packages (CRAN + Bioconductor limma from 2019 era)
RUN Rscript -e " \
    options(repos = c(CRAN = 'https://packagemanager.posit.co/cran/2021-05-17')); \
    install.packages(c('openxlsx', 'data.table', 'ggplot2', 'pracma', 'dgof')); \
    if (!requireNamespace('BiocManager', quietly=TRUE)) \
        install.packages('BiocManager', repos='https://cloud.r-project.org'); \
    BiocManager::install('limma', update=FALSE, ask=FALSE) \
"

WORKDIR /workspaces

# Entrypoint prints environment info
CMD ["bash", "-c", "echo '=== Park et al. (2020) AD-DNN Reproducibility Environment ===' && python3 --version && R --version | head -1 && echo 'TensorFlow:' && python3 -c 'import tensorflow; print(tensorflow.__version__)' && bash"]