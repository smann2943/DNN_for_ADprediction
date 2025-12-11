# Define a vector of package names to be installed
packages_to_install <- c("openxlsx", "data.table", "remotes", "pracma", "dgof", "BiocManager")

# Check if packages are installed and install them if they are not
# This loop iterates through each package name in the vector
for (pkg in packages_to_install) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load the installed packages
# The lapply function applies the library function to each package name
invisible(lapply(packages_to_install, library, character.only = TRUE))

BiocManager::install("limma")
# Your R script can now use functions from these packages
# For example:
# data <- read_csv("your_data.csv")
# ggplot(data, aes(x = column1, y = column2)) + geom_point()
