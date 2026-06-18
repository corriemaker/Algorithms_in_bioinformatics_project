# Comparison of non-nested versus true-nested cross-validation through SMM and ANN model evaluation
A terminal-based implementation of non-nested versus true-nested cross validation for a SMM (stabilised matrix method) and a ANN (non-linear artificial neural network). Predictive performance metrics are used to measure peptide-MHC binding models.

## Hypothesis

Non-nested ("fake") cross-validation will produce higher performance estimates than nested ("true") cross-validation for both models. This overestimation bias will be larger for the more complex ANN model than for the simpler SMM model.
The performance gap will diminish as the training dataset size increases, for both models.

## Hyperparameter Optimisation
For ANN:
Hidden layer size for model capacity
Epsilon, learning rate which controls gradient descent 
L2 regularization/weight decay to prevent overfitting
Early stopping if evaluation error plateaus 
Random seed 

For SMM:
Pseudocount value 
Stabilization parameter 

## Performance Metrics
AUC (Area Under the ROC Curve) with a 500 nM binding cutoff as the primary performance metric, consistent with the Peters et al. 2006 benchmark. As a secondary metric, the Pearson correlation coefficient (PCC) is reported. 

## Requirements
MHC Class I binding data alleles (n = x) sourced from "A Community Resource Benchmarking Predictions of Peptide Binding to MHC- I molecules" by Peters et al. 2006.
DOI: https://doi.org/10.1371/journal.pcbi.0020065

-Python 3.x or higher

## How to run

## File Overview
