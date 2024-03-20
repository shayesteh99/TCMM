# Topology-Constrained Metric Matching (TCMM)

This code is based on the method decribed in the paper **"Optimal Tree Metric Matching Enables Phylogenomic Branch Length Reconciliation"**(https://www.biorxiv.org/content/10.1101/2023.11.13.566962v1.full.pdf).

To use this tool use the following command to install **cxvpy**:

```
pip install cvxpy
```
## TCMM For multiple reference trees:
Use the following command to run TCMM for multiple reference trees. This code outputs a new tree and a set of branch lengths for each reference tree.

```
python multiple_tree_matching.py -i query_tree.trees -r reference_trees.trees -o out_file -l 0.001 -g log_file.txt
```

The passing arguments are:
>- `--input` or `-i`: The query tree in the newick format.
>- `--ref` or `-r`: The reference trees in the newick format one per line.
>- `--reg_coef` or `-l`: The regularization coeffiecient (lambda). Set lambda to 0 if the query tree does not have branch lengths. Higher lambda values conserve the original branch lengths of the query tree more. Default vallue is 0.001.
>- `--output_file ` or `-o`: The name of the output file without any extenstions. The code will produce three output files: "out_file.trees", "out_file.branches", and "out_file.rates". "out_file.trees" contains the new trees that have the same topology as the query tree and the branch lengths are the assgined branch lengths by TCMM that match the corresponding reference tree. "out_file.branches" and "out_file.rates" contain the lit of branch lengths and rates (new branch length devided by the original branch length), respectively. These values appear in the order of the post-order traversal of the query tree.
>- `--log_file` or `-g`: The log file contains the final loss of the optimization for matching to each reference tree. This argument is optional.
