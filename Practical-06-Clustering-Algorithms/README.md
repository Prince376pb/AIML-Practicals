# Practical 06 - Clustering Algorithms

## Practical Title
Implementation and Evaluation of Clustering Algorithms

## Problem Statement
Implement clustering algorithms on a dataset to identify meaningful groups within the data. Evaluate the clustering results using Silhouette Score, Davies-Bouldin Index, and visual inspection.

## Dataset
A dataset of 300 data points with two features is generated using Scikit-learn's `make_blobs` function.

## Algorithms Used
- K-Means Clustering
- Hierarchical Clustering

## Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index

## Technologies Used
- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## Results
Both algorithms generated four clusters.

| Algorithm | Silhouette Score | Davies-Bouldin Index |
|---|---:|---:|
| K-Means | 0.7582 | 0.3362 |
| Hierarchical Clustering | 0.7582 | 0.3362 |

Both algorithms obtained the same evaluation scores in this experiment.
