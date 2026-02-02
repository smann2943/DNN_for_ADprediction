##################################################################################################
# investigate_DEG_DMP.R
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 11/1/2018
# Date lastly modified: 7/1/2019
# Purpose: to perform DEG, DMP analysis with gene expression, DNA methylation dataset using Limma


closeAllConnections()
rm(list=ls())

#source("http://bioconductor.org/biocLite.R")
#biocLite(c("zip"))

library(openxlsx)
library(data.table)
library(parallel)
library(pracma)
library(dgof) 
library(limma)


## normalization by column sum (each column indicates sample)
normalization_by_colsum <- function(input_df){
  input_df = apply(input_df, 2, function(x){x/sum(x)})
  return(input_df)
}


## normalization by quantile
quantile_normalisation <- function(df){
  df_rank <- apply(df,2,rank,ties.method="min")
  df_sorted <- data.frame(apply(df, 2, sort))
  df_mean <- apply(df_sorted, 1, mean)
  
  index_to_mean <- function(my_index, my_mean){
    return(my_mean[my_index])
  }
  
  df_final <- apply(df_rank, 2, index_to_mean, my_mean=df_mean)
  rownames(df_final) <- rownames(df)
  return(df_final)
}

#' Perform differential expression / methylation analysis using limma
#'
#' doDEG runs a differential expression (DEG) or differential methylation (DMP)
#' analysis comparing Alzheimer's Disease (AD) vs Normal samples using limma.
#' The input TSV must contain a SampleID column and two rows named "Label_No"
#' and "Label_AD" that indicate sample labels. The function fits a linear model,
#' computes the "AD - Normal" contrast, extracts the top table (BH-adjusted p-values),
#' and applies decideTests with the provided p-value and log-fold-change thresholds.
#' Results are written to two TSV files in the current working directory.
#'
#' @param input_tsv character Path to input TSV (tab-delimited) with SampleID column.
#' @param lfc_threshold numeric Log-fold-change threshold passed to decideTests.
#' @param p_value_threshold numeric P-value threshold passed to decideTests.
#' @param k integer Fold index used to label output files.
#' @param type character Either "train" or "test", used in output filenames.
#' @param mode character Either "DEG" or "DMP", used in output filenames.
#' @return Invisibly returns NULL. Side effects: writes two TSV files:
#'   - "[<type> <k] AD <mode>.tsv" (topTable)
#'   - "[<type> <k] AD <mode> pval(... )_lfc(... ).tsv" (decideTests results)
#' @details The function uses limma::lmFit, limma::makeContrasts, limma::eBayes,
#'   limma::topTable and limma::decideTests. It expects label rows "Label_No"
#'   and "Label_AD" in the input; these are combined to form the design.
#' @examples
#' doDEG('../XY_gexp_train_1.tsv', lfc_threshold=1, p_value_threshold=0.01, k=1, type='train', mode='DEG')
#' @export
doDEG <- function(input_tsv, lfc_threshold, p_value_threshold, k, type, mode){
  ge_df <- read.csv(input_tsv, header=TRUE, sep="\t")
  rownames(ge_df)<-ge_df$SampleID
  
  print(head(ge_df[1:5, 1:5]))
  print(tail(ge_df[, 1:5], 5))
  
  ge_df_2<-ge_df[ , -which(names(ge_df) %in% c("SampleID"))]
  #print(head(ge_df_2[1:5, 1:5]))
  #print(tail(ge_df_2[, 1:5], 5))
  
  ge_df_2 <- as.data.frame(sapply(ge_df_2, as.numeric))
  rownames(ge_df_2)<-ge_df$SampleID
  print(head(ge_df_2[1:5, 1:5]))
  print(tail(ge_df_2[, 1:5], 5))
  
  
  No_label <- ge_df_2["Label_No", ]
  AD_label <- ge_df_2["Label_AD", ]
  
  AD_label_2 <- AD_label
  AD_label_2[AD_label_2 == '1'] <- 2
  
  label_df <- rbind(No_label, AD_label_2)
  label <- colSums(label_df)
  label_df <- rbind(label_df, label)
  rownames(label_df) <- c("Label_No", "Label_AD", "Label_NoAD")
  
  label_list <- unname(unlist(label_df["Label_NoAD",]))
  
  ## remove two row "Label_No", "Label_AD"
  drow <- c("Label_No", "Label_AD")
  ge_df_2_rem <- ge_df_2[!rownames(ge_df_2) %in% drow, ]
  
  print("before design matrix")
  print(head(ge_df_2_rem[1:5, 1:5]))
  print(tail(ge_df_2_rem[, 1:5], 5))
  
  
  ## Design matrix
  design <- model.matrix(~ 0 + factor(label_list))
  colnames(design)<-c("Normal", "AD")
  
  fit  <- lmFit(ge_df_2_rem, design)
  contrast.matrix_NoAD <- makeContrasts(AD-Normal, levels=design)
  
  fit2_CY <- contrasts.fit(fit, contrast.matrix_NoAD)
  fit2_CY <- eBayes(fit2_CY)
  
  ## output everything for contrast
  n <- nrow(ge_df_2)
  
  ## get p-value by BH
  #' Extract Top Differentially Expressed/Methylated Genes
  #'
  #' Retrieves the top n genes/probes ranked by statistical significance from a 
  #' linear model comparison between Alzheimer's Disease (AD) and Normal control groups.
  #'
  #' @details
  #' This function applies the Benjamini-Hochberg multiple testing correction to control
  #' the false discovery rate (FDR). The results are sorted by adjusted p-value, with
  #' the most significantly differential genes/probes appearing first.
  #'
  #' @param fit2_CY A fitted linear model object (likely from limma package) containing
  #'   the results of differential expression/methylation analysis
  #' @param coef Character string specifying the contrast to extract: "AD - Normal"
  #'   compares the AD group relative to the Normal control group
  #' @param n Integer specifying the number of top results to return
  #' @param adjust Character string specifying the p-value adjustment method.
  #'   "BH" uses Benjamini-Hochberg FDR correction
  #'
  #' @return
  #' A data frame containing the top n genes/probes with columns including:
  #' logFC (log fold change), AveExpr (average expression), t-statistic, 
  #' P.Value, adj.P.Val (adjusted p-value), and B-statistic
  #'
  #' @note
  #' Results are ordered by statistical significance (adjusted p-value).
  #' More significant genes appear at the beginning of the returned table.
  top1 <- topTable(fit2_CY, coef="AD - Normal", n=n, adjust = "BH")
  
  out_tsv <- sprintf('./[%s %s] AD %s.tsv', type, k, mode)
  cat("Writing topTable to file:", out_tsv, "\n")
  cat("Arguments: coef='AD - Normal', n=", n, ", adjust='BH'\n", sep="")
  
  # print out the columns headings and first few rows of top1 for verification
  print(head(top1))
  
  
  
  write.table(top1, out_tsv, sep="\t", row.names=T)
  
  
  #' Perform Differential Expression Analysis
  #'
  #' This function applies a statistical test to determine which features (genes) are differentially expressed 
  #' based on the fitted model `fit2_CY`. It uses specified thresholds for p-value and log-fold change (lfc) 
  #' to decide which features are considered significant.
  #'
  #' @param fit2_CY A fitted model object containing the results of the differential expression analysis.
  #' @param p_value_threshold A numeric value representing the threshold for p-values. Features with p-values 
  #'                         below this threshold are considered significant.
  #' @param lfc_threshold A numeric value representing the threshold for log-fold change. Features with 
  #'                      absolute log-fold change values above this threshold are considered significant.
  #'
  #' @return A result object indicating which features are significantly differentially expressed based on 
  #'         the specified thresholds.
  #'
  #' @examples
  #' # Example usage:
  #' res <- decideTests(fit2_CY, p.value=0.05, lfc=1)
  #'
  #' @seealso \code{\link{fit2_CY}}, \code{\link{decideTests}}
  
  
  # TODO:  evaluate what decideTests is returning here and how it impacts the results of the DNN
  res <- decideTests(fit2_CY, p.value=p_value_threshold, lfc=lfc_threshold)
  summary(res)
  out_tsv <- sprintf('./[%s %s] AD %s pval(%4.2f)_lfc(%3.1f).tsv', type, k, mode, p_value_threshold, lfc_threshold)
  write.table(res, out_tsv, sep="\t", row.names=T)
  
}



################################################################################################
setwd("D:/Development/ADprediction_git/ADprediction/code/02 feature selection")
dir.create("../../results/k_fold_train_test/DEG", recursive = TRUE)
setwd("../../results/k_fold_train_test/DEG")

p_value_threshold=0.01
lfc_threshold=1

input_tsv <- '../XY_gexp_train_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "train", "DEG")
input_tsv <- '../XY_gexp_train_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "train", "DEG")
input_tsv <- '../XY_gexp_train_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "train", "DEG")
input_tsv <- '../XY_gexp_train_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "train", "DEG")
input_tsv <- '../XY_gexp_train_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "train", "DEG")

input_tsv <- '../XY_gexp_test_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "test", "DEG")
input_tsv <- '../XY_gexp_test_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "test", "DEG")
input_tsv <- '../XY_gexp_test_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "test", "DEG")
input_tsv <- '../XY_gexp_test_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "test", "DEG")
input_tsv <- '../XY_gexp_test_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "test", "DEG")

################################################################################################
setwd("../../../")
dir.create("./results/k_fold_train_test/DMP", recursive = TRUE)
setwd("./results/k_fold_train_test/DMP")

p_value_threshold=0.01
lfc_threshold=0.58

input_tsv <- '../XY_meth_train_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "train", "DMP")
input_tsv <- '../XY_meth_train_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "train", "DMP")
input_tsv <- '../XY_meth_train_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "train", "DMP")
input_tsv <- '../XY_meth_train_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "train", "DMP")
input_tsv <- '../XY_meth_train_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "train", "DMP")

input_tsv <- '../XY_meth_test_1.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 1, "test", "DMP")
input_tsv <- '../XY_meth_test_2.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 2, "test", "DMP")
input_tsv <- '../XY_meth_test_3.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 3, "test", "DMP")
input_tsv <- '../XY_meth_test_4.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 4, "test", "DMP")
input_tsv <- '../XY_meth_test_5.tsv'
doDEG(input_tsv, lfc_threshold, p_value_threshold, 5, "test", "DMP")

