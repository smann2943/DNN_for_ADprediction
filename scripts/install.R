# 1. Setup Mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# 2. Bootstrap 'remotes'
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", dependencies = TRUE)
}

# 3. MANUALLY PIN Rcpp (The Culprit)
# Rcpp 1.0.3 was the standard for late 2019 and is highly compatible with R 3.5.2
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  message("Installing legacy Rcpp 1.0.3...")
  remotes::install_version("Rcpp", version = "1.0.3", upgrade = "never")
}

# 4. Install openxlsx 4.2.3
if (!requireNamespace("openxlsx", quietly = TRUE)) {
  message("Installing openxlsx 4.2.3...")
  remotes::install_version("openxlsx", version = "4.2.3", upgrade = "never")
}

# 5. Define and Install remaining packages
packages_to_install <- c("data.table", "pracma", "dgof", "BiocManager")
for (pkg in packages_to_install) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg == "BiocManager") {
      remotes::install_version("BiocManager", version = "1.30.4", upgrade = "never")
    } else {
      install.packages(pkg, dependencies = TRUE)
    }
  }
}

# 6. Bioconductor
if (!requireNamespace("limma", quietly = TRUE)) {
  BiocManager::install("limma", version = "3.8", update = FALSE, ask = FALSE)
}

# 7. Final Loading Verification
all_packages <- c("Rcpp", "data.table", "pracma", "dgof", "BiocManager", "openxlsx", "limma")
for (pkg in all_packages) {
  if (!require(pkg, character.only = TRUE)) {
    stop(paste("Failed to load:", pkg))
  }
}

message("Full environment successfully loaded.")