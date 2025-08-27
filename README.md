# Japanese MLSE Statistical Analysis Code

## Overview

This repository contains the statistical analysis code used in the study "Validation of the Japanese MLSE for primary progressive aphasia: a multicenter study". The analysis was conducted using Bayesian methods to assess the Japanese version of the Mini-Linguistic State Examination (MLSE) for classifying primary progressive aphasia (PPA) subtypes.

## Repository Contents

The repository includes the following scripts:

- **`multinomial_bayesian_logistic.py`**: Bayesian multinomial logistic regression for three-way classification among PPA subtypes (naPPA, svPPA, lvPPA).
- **`binomial_bayesian_logistic.py`**: Bayesian logistic regression for binary classification (HP vs each PPA subtype).
- **`cutoff_from_HP.py`**: Derivation of domain-specific cutoff values from healthy participant distributions using 95% HPD intervals.
- **`decision_tree.py`**: Diversity-constrained CART with Bayesian bootstrap for clinical decision rules.
- **`supplementary_binomial_frequentist_logistic.py`**: Frequentist logistic regression for methodological continuity with prior studies.
- **`supplementary_rf.py`**: Random forest classification with nested cross-validation for supplementary analysis.
- **`requirements.txt`**: Exact versions of all Python dependencies for reproducibility.

## Reproducibility

For full reproducibility:
- Python 3.10.12 with exact package versions listed in `requirements.txt`
- Key packages: pandas 2.3.1, NumPy 2.1.3, SciPy 1.15.3, statsmodels 0.14.5, scikit-learn 1.7.1, PyMC 5.25.1, ArviZ 0.22.0
- Statistical methods and results are detailed in our published manuscript

## Citation

If you use this code or model in your research, please cite our paper:

```text
[Our Paper Title]
[Our Authors]
[Journal Name, Year]
[DOI or URL]
```
*Citation details will be added after the publication of our paper.*


## Note

- The repository does not include raw patient data due to privacy and ethical considerations.
- The statistical models were implemented using Bayesian inference with PyMC 5.25.1.
- The scripts assume that data preprocessing (e.g., MLSE scoring, diagnostic classification) has been completed according to the Japanese MLSE manual.
- If you use this code, please cite the corresponding paper.

## License

This code is released under the MIT License.
