# DNN_for_ADprediction

Predict Alzheimer's Disease (AD) by integrating gene expression and DNA methylation using Deep Neural Networks and conventional machine learning models.

This repository implements data preprocessing, feature selection, hyper-parameter search and prediction experiments (DNN and ML) used in the original study. The code is written in Python (and a small R script for DEG/DMP investigation) and operates on tabular datasets included in `dataset/`.

## Overall algorithm

[![Overall algorithm workflow](https://user-images.githubusercontent.com/34843393/61431751-8801f280-a969-11e9-87b3-aa3d2b569abc.PNG)](https://www.sciencedirect.com/science/article/abs/pii/S0957417419305834)

Click the image to open the published article (ScienceDirect). See the "Background" section for a short summary and citation.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start / Usage](#quick-start--usage)
	- [Prepare data](#prepare-data)
	- [Preprocessing](#preprocessing)
	- [Feature selection](#feature-selection)
	- [Hyper-parameter search](#hyper-parameter-search)
	- [Run prediction (DNN / ML)](#run-prediction-dnn--ml)
- [Development notes](#development-notes)
- [Contributing](#contributing)
- [Tests](#tests)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#acknowledgements)

- [Devcontainer (local development)](#devcontainer-local-development)
- [GPU devcontainer (NVIDIA)](#gpu-devcontainer-nvidia)

## Background

The project builds models that predict Alzheimer's Disease status by integrating two omics modalities: gene expression and DNA methylation. The implementation includes several dimension-reduction / feature-selection strategies (DEG/DMP, PCA, t-SNE), classical ML models (Random Forest, SVM, Naive Bayes) and a deep neural network model.

Paper summary

This codebase implements and reproduces the approach from the published study (Park et al., Expert Systems with Applications, 2020) which proposes an integrative multi-omics prediction framework for Alzheimer's Disease (https://www.sciencedirect.com/science/article/abs/pii/S0957417419305834).

- Goal: combine gene expression and DNA methylation profiles to improve AD classification compared with single-omics models.
- Approach: perform initial feature selection (DEG for gene expression, DMP/DMG for methylation), optionally reduce dimensionality (PCA, t-SNE), then integrate the two modalities into a single input for downstream classifiers. The study evaluates both conventional ML classifiers (Random Forest, SVM, Naive Bayes) and a deep neural network architecture.
- Optimization & evaluation: hyper-parameters are tuned via Bayesian optimization and models are evaluated using K-fold cross-validation. The paper reports that integrating the two data types and using the DNN architecture improved predictive performance compared to single-modality baselines (see the paper for detailed performance metrics and experimental settings).

Reference (paraphrased): Park, C., et al., "Prediction of Alzheimer's disease by integrating gene expression and DNA methylation using deep neural networks", Expert Systems with Applications (2020). https://doi.org/10.1016/j.eswa.2019.112888 (ScienceDirect: S0957417419305834)

## Features

- Scripts for splitting and preparing K-fold datasets.
- Feature selection using DEG (differentially expressed genes) and DMP/DMG (differentially methylated probes/genes).
- Hyper-parameter optimization using Bayesian optimization.
- Prediction pipelines: DNN and conventional ML with integrated datasets.

## Requirements

- Python 3.10+ is recommended (code was originally developed for older TensorFlow versions; current `requirements.txt` pins modern binaries).
- Install Python dependencies from `requirements.txt`:

	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

Notes:
- The repository contains `requirements.txt` with pinned versions used here. If you need a GPU-enabled TensorFlow or a specific TF1 behavior, adjust `requirements.txt` accordingly. Some scripts use legacy TF1 APIs; parts of the code have been left as-is to preserve the original behavior.

## Devcontainer (local development)

This repository supports development inside a VS Code devcontainer. Using the devcontainer ensures consistent tooling and versions and makes it easier to reproduce experiments.

Prerequisites:
- Docker installed and running on the host.
- VS Code with the "Remote - Containers" (Dev Containers) extension (or the built-in devcontainer support in newer VS Code releases).

Quick steps:
1. In VS Code: use "Remote-Containers: Open Folder in Container" (Command Palette) and select the repository root.
2. The devcontainer will build and open the workspace inside the container. Follow the status messages in VS Code for build progress.

Notes:
- The devcontainer is configured for Ubuntu-based images (this workspace uses Ubuntu 24.04.2 LTS in the dev environment). If you modify devcontainer settings, rebuild the container (Command Palette: "Dev Containers: Rebuild Container").
- The devcontainer provides a consistent Python environment; you may still want to create and activate a virtual environment inside the container as described in the "Requirements" section.

## GPU devcontainer (NVIDIA)

If you plan to run GPU-accelerated training (TensorFlow / PyTorch), the host must provide NVIDIA GPU support and the NVIDIA Container Toolkit must be installed so containers can access GPUs.

Host prerequisites:
- NVIDIA GPU with drivers installed on the host (choose a driver version compatible with your CUDA and framework versions).
- Docker Engine installed on the host.

Install NVIDIA Container Toolkit (summary):

- Follow the official NVIDIA Container Toolkit installation guide for Ubuntu/Debian for exact, up-to-date commands and repository steps:

	https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian

- The guide shows how to add the NVIDIA apt repository and install the toolkit. A typical sequence you would run on the host (check the official guide for the recommended and current commands) looks like:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
# Add the NVIDIA package repository (follow the official guide's commands)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

- After installation, verify GPUs are accessible to Docker on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu24.04 nvidia-smi
```

Using the GPU-enabled devcontainer:
- Ensure your devcontainer configuration passes GPU access to the container. This can be achieved by adding runtime arguments or using the `--gpus` option when launching the container. In `devcontainer.json`, you can include Docker `runArgs` such as:

```json
"runArgs": ["--gpus", "all"]
```

- Rebuild and reopen the devcontainer after changing `devcontainer.json`.

Verify GPU availability inside the container:

```bash
# inside the container
nvidia-smi
# or in Python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Troubleshooting notes:
- If `nvidia-smi` is not available inside the container but works on the host, ensure the toolkit is installed on the host and that the container was launched with GPU access (`--gpus all` or equivalent runtime). Restart Docker after installing the toolkit.
- If drivers mismatch or CUDA toolkit compatibility errors occur, check the host driver version and select a matching CUDA base image for your containerized workloads.

If you want, I can add a sample `devcontainer.json` and Dockerfile tuned for GPU usage (small, safe defaults) — tell me whether you prefer a CPU-only devcontainer or a GPU-enabled devcontainer and I'll add the files.

## Repository layout

Top-level:

- `code/` - all analysis scripts, divided into numbered stages:
	- `01 data preprocessing/` - scripts to split and format input data
	- `02 feature selection/` - feature selection and annotation (includes an R script)
	- `03 hyperparameter search/` - Bayesian optimization routine
	- `04 prediction/` - ML and DNN prediction pipelines (`AD_Prediction_ML.py`, `AD_Prediction_DNN.py`)
- `dataset/` - example input tables used by the code (e.g. `allforDNN_ge_sample.tsv`, `allforDNN_me_sample.tsv`, `DEG_list.tsv`, `DMP_list.tsv`)
- `scripts/` - helper scripts (e.g. to download GPL mapping file)

## Quick start / Usage

The examples below assume you are in the repository root and have activated your Python virtual environment.

### Prepare data

1. Put your input files in `dataset/` or update paths used by the scripts. The repository includes example files:

	- `dataset/allforDNN_ge_sample.tsv` — gene expression matrix (samples x genes)
	- `dataset/allforDNN_me_sample.tsv` — DNA methylation matrix (samples x CpG probes)
	- `dataset/DEG_list.tsv` — DEG results (Limma or equivalent)
	- `dataset/DMP_list.tsv` — DMP/DMG results

2. A platform mapping file `GPL13534-11288.txt` is required by some scripts (used to map probe IDs to gene symbols). You can download it with the helper script:

	python scripts/download_GPL13534-11288.py

This will create `dataset/GPL13534-11288.txt` (check the helper script for the exact behavior).

### Preprocessing

Run the preprocessing to split input data into train/test / K-folds (script prints options using --help if available):

	python "code/01 data preprocessing/Split_Inputdata.py"

If the script accepts arguments, pass input and output folders as needed. The default paths used in later scripts expect a `results/k_fold_train_test` directory structure.

### Feature selection

The feature selection step includes an R script to inspect DEG/DMP and a Python script to annotate DMPs. To run the R script (requires R installed):

	Rscript "code/02 feature selection/01 investigate_DEG_DMP.R"

Then annotate DMPs (Python):

	python "code/02 feature selection/02 Annotate_DMP.py"

Check script headers and help text for any required flags or input paths.

### Hyper-parameter search

Run the Bayesian hyper-parameter search which integrates datasets and performs K-fold evaluation:

	python "code/03 hyperparameter search/BayesianOpt_HpParm_Search.py"

This script consumes the example files in `dataset/` and writes logs/results under `results/k_fold_train_test_results/` by default.

### Run prediction (DNN / ML)

Two main prediction entry points are provided:

- Conventional ML experiments:

	python "code/04 prediction/AD_Prediction_ML.py" --input ./results/k_fold_train_test --output ./results/k_fold_train_test_results

- Deep Neural Network experiments:

	python "code/04 prediction/AD_Prediction_DNN.py" --input ./results/k_fold_train_test --output ./results/k_fold_train_test_results

Both scripts accept `--input` and `--output` arguments with the shown defaults. The input directory should contain per-fold files produced during preprocessing and feature-selection steps (the scripts expect a particular TSV layout — see the script docstrings and prints for column expectations). Example paths used inside the scripts:

- `input_dir/XY_gexp_train_1_ML_input.tsv`
- `input_dir/XY_meth_train_1_ML_input.tsv`
- `input_dir/DEG/[train 1] AD DEG.tsv`
- `input_dir/DMP/[train 1] AD DMP.tsv`

If you run into missing-file errors, inspect `code/04 prediction/AD_Prediction_DNN.py` and `AD_Prediction_ML.py` to see the path templates.

## Development notes

- Many scripts were originally written for older TensorFlow (1.x) and may reference `tf.placeholder`, `tf.Session()` and other TF1 APIs. The `requirements.txt` currently pins TensorFlow 2.x; if you need TF1 semantics, either install a TF1-compatible version (e.g. `tensorflow==1.15`) or run the code with `tf.compat.v1` wrappers (some scripts already use `tf.compat.v1`).
- The code mixes styles and contains some legacy numpy / pandas calls (e.g. `as_matrix()`); upgrading to modern pandas APIs may be necessary for long-term maintenance.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-change`.
3. Add tests for any behavioral changes.
4. Run lint/tests locally.
5. Submit a pull request with a clear description of your change.

Please open an issue before large refactors so we can discuss design.

## Tests

This repository doesn't include automated unit tests at the moment. To do a quick smoke test, run the scripts with `--help` or run a single small end-to-end flow on a subset of data (create a tiny `dataset/` table with a few rows to confirm the scripts execute).

Example smoke run (prints help / quick-run behavior):

	python "code/04 prediction/AD_Prediction_DNN.py" --help

## Authors

- Chihyun Park — original author (contact: chihyun.park@yonsei.ac.kr)

Repository maintained by: smann2943 (GitHub)

## License

No license file is included in this repository. If you intend to open-source this project, add a LICENSE (for example MIT, BSD, or Apache-2.0) in the repository root. Until a license is added, usage is governed by the repository owner's terms.

## Acknowledgements

- Original implementation and dataset authorship: see code headers and script metadata.
- The project references standard tools such as scikit-learn, TensorFlow and the bayesian-optimization library.

