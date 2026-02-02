# Datasets

## Overview

This document provides an overview of the datasets used in the [DNN for AD Prediction](https://github.com/smann2943/DNN_for_ADprediction) repository. The repository focuses on predicting Alzheimer's Disease (AD) status by integrating gene-expression and DNA-methylation profiles using deep neural networks (DNN) and conventional machine learning models. The datasets are primarily tabular files stored in the `dataset/` directory and are processed through preprocessing, feature selection, hyperparameter tuning, and prediction pipelines.

The core datasets consist of omics data (gene expression and DNA methylation) along with feature selection results (differentially expressed genes - DEG, and differentially methylated probes/genes - DMP/DMG). These are used to train classifiers for AD prediction via K-fold cross-validation.

**Key Datasets:**
- Gene expression matrix
- DNA methylation matrix
- DEG list
- DMP list
- Platform mapping file (for annotation)

All datasets are provided in TSV (tab-separated values) format for easy loading with Python libraries like pandas.

## Gene Expression Dataset

### Description
The gene expression dataset (`allforDNN_ge_sample.tsv`) contains RNA microarray data measuring the expression levels of genes across samples. This data is derived from brain tissue samples of individuals with and without AD. It serves as one omics modality for input to the models, capturing transcriptional changes associated with AD pathology.

- **Source:** Likely from public repositories like GEO (Gene Expression Omnibus), processed for AD prediction.
- **Samples:** Rows represent samples (individuals); columns represent genes.
- **Labels:** AD status (e.g., control vs. AD) is typically encoded in sample metadata or as a separate column.
- **Dimensions:** Samples × Genes (exact counts depend on the processed data; typically hundreds of samples and thousands of genes).

### Structure
- **File Format:** TSV
- **Columns:** Gene symbols/IDs as headers; expression values (e.g., log2-normalized intensities).
- **Rows:** Sample IDs or indices.
- **Example Snippet:**
  ```
  Sample_ID    Gene1    Gene2    ...    GeneN
  Sample1      5.2      3.1      ...    4.8
  Sample2      6.1      2.9      ...    5.3
  ...
  ```

### Usage in the Repository
- **Preprocessing (`code/01 data preprocessing/Split_Inputdata.py`):** Loaded via pandas, split into train/test sets and K-fold partitions. Outputs per-fold files (e.g., `XY_gexp_train_1_ML_input.tsv`).
- **Feature Selection (`code/02 feature selection/`):** Filtered using DEG list to retain only differentially expressed genes.
- **Integration and Modeling:** Combined with methylation data post-feature selection. Fed into classifiers in `AD_Prediction_ML.py` and `AD_Prediction_DNN.py` for AD prediction. Optional dimensionality reduction (PCA/t-SNE) may be applied.
- **Requirements:** Ensure the file is placed in `dataset/`. Scripts handle loading and normalization as needed.

## DNA Methylation Dataset

### Description
The DNA methylation dataset (`allforDNN_me_sample.tsv`) contains epigenomic data measuring methylation levels at CpG sites across samples. This captures DNA methylation changes linked to AD, providing complementary information to gene expression.

- **Source:** Likely from Illumina methylation arrays (e.g., 450K or EPIC), processed for AD studies.
- **Samples:** Rows represent samples; columns represent CpG probes.
- **Labels:** Same as gene expression (AD status).
- **Dimensions:** Samples × CpG Probes (typically hundreds of samples and tens of thousands of probes).

### Structure
- **File Format:** TSV
- **Columns:** Probe IDs (e.g., cg00000029) as headers; beta-values (0-1 methylation levels).
- **Rows:** Sample IDs or indices.
- **Example Snippet:**
  ```
  Sample_ID    cg0001    cg0002    ...    cgN
  Sample1      0.45      0.72      ...    0.31
  Sample2      0.52      0.68      ...    0.29
  ...
  ```

### Usage in the Repository
- **Preprocessing:** Similar to gene expression; split into K-fold files (e.g., `XY_meth_train_1_ML_input.tsv`).
- **Feature Selection:** Filtered using DMP list to select differentially methylated probes/genes. Annotated with `02 Annotate_DMP.py` using the GPL mapping file.
- **Integration and Modeling:** Merged with filtered gene expression data to form multi-omics input for Random Forest, SVM, Naive Bayes, and DNN models.
- **Requirements:** Place in `dataset/`. Annotation requires the GPL mapping file.

## Differentially Expressed Genes (DEG) List

### Description
The DEG list (`DEG_list.tsv`) contains results from differential expression analysis (e.g., using Limma in R), identifying genes significantly upregulated or downregulated in AD vs. control samples.

- **Source:** Generated from the gene expression dataset using statistical tests.
- **Features:** Genes with p-value < threshold (e.g., adjusted p < 0.05) and log-fold change.

### Structure
- **File Format:** TSV
- **Columns:** Gene ID/symbol, logFC, AveExpr, t-statistic, P.Value, adj.P.Val, etc.
- **Rows:** Selected DEG entries.
- **Example Snippet:**
  ```
  Gene    logFC    AveExpr    P.Value    adj.P.Val
  APP     1.23     5.67       0.001      0.045
  ...
  ```

### Usage in the Repository
- **Feature Selection (`code/02 feature selection/01 investigate_DEG_DMP.R`):** Inspected and filtered for model input.
- **Modeling:** Used to subset the gene expression matrix, reducing dimensionality before integration.
- **Requirements:** Generated or provided in `dataset/`.

## Differentially Methylated Probes/Genes (DMP) List

### Description
The DMP list (`DMP_list.tsv`) identifies CpG probes or associated genes with significant methylation differences in AD samples.

- **Source:** From differential methylation analysis on the methylation dataset.
- **Features:** Probes/genes with significant beta-value differences.

### Structure
- **File Format:** TSV
- **Columns:** Probe ID, associated gene, delta-beta, p-value, etc.
- **Rows:** Selected DMP entries.
- **Example Snippet:**
  ```
  Probe    Gene    deltaBeta    P.Value
  cg0001   ABC1    -0.15        0.002
  ...
  ```

### Usage in the Repository
- **Feature Selection (`code/02 feature selection/02 Annotate_DMP.py`):** Annotated to gene level using GPL file; inspected in R script.
- **Modeling:** Subsets the methylation matrix for integration.
- **Requirements:** Provided in `dataset/`.

## Platform Mapping File

### Description
The GPL mapping file (`GPL13534-11288.txt`) maps methylation probe IDs to gene symbols, enabling annotation of DMPs to biological features.

- **Source:** Downloaded from NCBI GEO platform GPL13534 (Illumina HumanMethylation450).
- **Purpose:** Bridge between probe-level data and gene-level interpretation.

### Structure
- **File Format:** TXT/TSV
- **Columns:** Probe ID, Gene Symbol, etc.
- **Rows:** Mappings for all probes.

### Usage in the Repository
- **Download (`scripts/download_GPL13534-11288.py`):** Script to fetch if missing.
- **Annotation (`code/02 feature selection/02 Annotate_DMP.py`):** Used to annotate DMP list.
- **Requirements:** Run the download script to populate `dataset/`.

## Getting Started with Datasets

1. **Download/Place Files:** Ensure all TSV files are in `dataset/`. Run `python scripts/download_GPL13534-11288.py` for the mapping file.
2. **Preprocessing:** Execute `python code/01 data preprocessing/Split_Inputdata.py` to generate K-fold splits.
3. **Feature Selection:** Run R script `code/02 feature selection/01 investigate_DEG_DMP.R` and Python `code/02 feature selection/02 Annotate_DMP.py`.
4. **Modeling:** Use outputs in `code/04 prediction/` scripts for ML/DNN training.
5. **Dependencies:** See `requirements.txt` for Python packages (e.g., pandas, scikit-learn, TensorFlow).

For more details on the workflow, refer to the repository [README](https://github.com/smann2943/DNN_for_ADprediction). If you have questions about data sourcing or extensions, check the original omics studies cited in the DEG/DMP analyses.

---

*Last updated: November 26, 2025*