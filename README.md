# Epic sklearn &mdash; An expansion pack for scikit-learn
[![Epic-sklearn CI](https://github.com/Cybereason/epic-sklearn/actions/workflows/ci.yml/badge.svg)](https://github.com/Cybereason/epic-sklearn/actions/workflows/ci.yml)

## What is it?
The **epic-sklearn** Python library is a companion to the scikit-learn library for machine learning.
It provides additional components and utilities, which can make working within the scikit-learn
framework even more convenient and productive.

The main difference in base-assumptions between scikit-learn and epic-sklearn is that epic-sklearn
has pandas as a dependency. Moreover, most epic-sklearn components support pandas objects (`DataFrame`
and `Series`) and "pass along" as much information as possible. For example, in most transformers,
if the features matrix is provided as a `DataFrame`, the transformed matrix will also be a `DataFrame`,
and the `index` (and `columns`, if applicable) will be preserved. There are also a few components specifically
designed for working only with pandas objects.


## Content Highlights
- **composite:** Classifiers acting on other classifiers.
- **feature_selection:**
  - **mutual_info:** Calculation of conditional mutual information between a feature and the target given another 
  feature, and feature selection algorithms based on conditional mutual information.
- **metrics:**
  - Metrics and scores for evaluating classification results and other data sets.
  - Also includes the **leven** module, allowing parallel computation of pairwise Levenshtein distances 
  between python strings.
- **neighbors:** Utilities relevant for nearest neighbors algorithms.
- **pipeline:** Transformers for constructing transformation pipelines.
  - Contains a transformer that splits
  the samples based on a criterion, and applies different transformations on each sample group.
- **plot:** Plotting utilities.
- **preprocessing:**
  - **categorical:** Transformers for encoding and processing categorical data.
  - **data:** Transformers for binning and manipulating data distribution. Includes the
  Yeo&ndash;Johnson transformation.
  - **general:** General-purpose transformers (e.g. select DataFrame columns, apply a function in parallel, 
  generate features from an iterator).
  - **label:** Utilities for encoding labels.
- **utils:** 
  - **data:** Generate random batches from data.
  - **general:** Functions for input validation and normalization.
  - **kneedle:** Implementation of the "Kneedle in a Haystack" algorithm.
  - **thresholding:** A helper for setting and applying a threshold based on classification metrics.


## Contributors
Thanks to [Yaron Cohen](https://github.com/cr-yaroncohen) for his contribution to this project.
