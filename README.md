# FWP Brain Volume Clustering

This project explores unsupervised grouping of brain MRI-derived volumes by learning compact latent features and clustering subjects in that latent space.

## Purpose

The goal is to identify meaningful structural patterns across subjects without using diagnosis labels as training targets. The workflow combines:

- Autoencoder-based feature learning from preprocessed brain volume data.
- Dimensionality reduction for visualization.
- Clustering (including spectral clustering and k-means) for subgroup discovery.
- Post-cluster analysis and sample export for visual inspection.

## Files and Folders

- `autoencoder_main.ipynb`: trains/uses the autoencoder and produces latent representations.
- `k-means.ipynb`: clustering experiments and metric comparison in reduced feature space.
- `cluster_analysis.py`: summarizes cluster composition and saves analysis artifacts.
- `export_spectral_cluster_samples.py`: exports representative examples from spectral clusters.
- `outputs/`: model files, latent arrays, PCA embeddings, and clustering CSV results.
- `cluster_analysis_results/`: generated analysis tables and summaries.

## Data And Outputs

- Input metadata is tracked in `metadata.csv`.
- Processed/test imaging data are under `fwp_brain_volume_clustering_test_data/`.
- Generated files are written mainly to `outputs/` and `cluster_analysis_results/`.