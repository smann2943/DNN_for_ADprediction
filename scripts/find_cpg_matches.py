#!/usr/bin/env python3
"""
Find all IDs where a specified gene is in the UCSC_RefGene_Name column in the GPL13534-11288.txt file
and determine if the ID column from the GPL file matches columns in the allforDNN_me_sample.tsv file.
Report the number of rows/samples for each match.

WHY: To identify CpG sites associated with a specific gene for methylation analysis.  The issue we were hitting is when running the DNN model, 
we got an error that there was no overlap between the DMG and DEG sets. This can be seen in the code here:
```[python]
degSet, _ = load_DEG_DMG(input_dir + "/DEG/[train " + str(k) + "] AD DEG.tsv", thresh_lfc_ge, thresh_pval_ge, "DEG", mapTableFile)
dmgSet, geneCpgSet_map = load_DEG_DMG(input_dir + "/DMP/[train " + str(k) + "] AD DMP.tsv", thresh_lfc_me, thresh_pval_me, "DMP", mapTableFile)

its_geneSet = degSet & dmgSet
    # Debugging info: show sizes of selected feature sets
print("degSet size: {}\tdmgSet size: {}\tintersection size: {}".format(len(degSet), len(dmgSet), len(its_geneSet)))		
```

The issue was found that the dmgSet had a small size.  The associated paper called out specific genes of interest, including CHM.  This script helps identify if there are CpG sites associated with specific genes in the methylation dataset.

Usage: python find_cpg_matches.py [GENE_NAME]
       Default gene: CHM
"""

import sys
import pandas as pd

# Get gene name from command line argument or use default
gene_name = sys.argv[1].upper() if len(sys.argv) > 1 else "CHM"

# Read the GPL file
print("Reading GPL13534-11288.txt...")
gpl_file = "dataset/GPL13534-11288.txt"
gpl_df = pd.read_csv(gpl_file, sep="\t", comment="#")

# Find all rows where the gene is in the UCSC_RefGene_Name column
print(f"Filtering for rows with {gene_name} in UCSC_RefGene_Name column...")
gene_ids = gpl_df[gpl_df['UCSC_RefGene_Name'].astype(str).str.split(';').str[0].str.fullmatch(gene_name, na=False, case=False)]['ID'].unique()
print(f"Found {len(gene_ids)} unique IDs with {gene_name} in UCSC_RefGene_Name")
print(f"{gene_name} IDs: {gene_ids.tolist()}")

# Read the methylation file - only header first to get column names
print("\nReading column headers from allforDNN_me_sample.tsv...")
me_file = "dataset/allforDNN_me_sample.tsv"
me_df = pd.read_csv(me_file, sep="\t", nrows=1)
me_columns = me_df.columns.tolist()

# Remove non-probe columns
exclude_cols = ['SampleID', 'Label_No', 'Label_AD']
me_probe_columns = [col for col in me_columns if col not in exclude_cols]

print(f"Found {len(me_probe_columns)} probe columns in methylation file")

# Find matches between gene IDs and methylation file columns
print(f"\nMatching {gene_name} IDs with methylation file columns...")
matches = []
for gene_id in gene_ids:
    if gene_id in me_probe_columns:
        matches.append(gene_id)
        print(f"  Found match: {gene_id}")

print(f"\n{'='*60}")
print(f"RESULTS for {gene_name}:")
print(f"{'='*60}")
print(f"Total IDs with {gene_name} in GPL file: {len(gene_ids)}")
print(f"Matching IDs found in methylation file: {len(matches)}")

if matches:
    print(f"\nMatching {gene_name} probe IDs in methylation file:")
    for match in matches:
        print(f"  - {match}")
else:
    print(f"\nNo matching {gene_name} IDs found in methylation file columns")