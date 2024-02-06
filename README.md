# Bilinear_Model

This repository hosts the implementation and applications of a bilinear model designed for analyzing neuronal gene expression and connectivity data. The work supports the findings presented in:

Qiao, M. (2023+). Deciphering the Genetic Code of Neuronal Type Connectivity: A Bilinear Modeling Approach.

## Overview
Leveraging concepts from recommendation systems, the bilinear model projects gene expressions of presynaptic and postsynaptic neuronal types into a shared latent space. This process reconstructs a cross-correlation matrix that closely approximates the neuronal connectivity matrix, bridging single-cell sequencing data with connectomic data of neuronal types to unravel the genetic underpinnings of specific connectivities.

## Getting Started
To explore the bilinear model applications, clone this repository and navigate to the desired application directory. Each directory contains the necessary code and documentation for applying the bilinear model to its respective dataset.

## Application to C. elegans

Implementation of the bilinear model for the C. elegans dataset, including comparative analysis against the SCM.

### Structure:
We used the same C. elegans data tested by the SCM (https://github.com/kpisti/SCM/tree/v1.0), including the innexin expression data (INXExpressionJustContact.csv), gap junction connectivity data (GapJunctContact.csv), and physical contact matrix (ContactSubgraphMatrix.csv). The notebook SCM_plots.ipynb replicates the SCM and generates relevant plots.

- bilinear_model_single_cells.py contains the source code of the bilinear model (scenario 1) for the C. elegans dataset.
- bilinear_model_single_cells_cv.ipynb performs 5-fold cross-validation, saving results in the folder named "cv_results".
- bilinear_model_single_cells_plot.ipynb generates figures in the paper.
- bilinear_model_single_cells_genetic_rules compares the genetic interactions from the bilinear model and those from the SCM.

## Application to Mouse Retina
Tailored implementation of the bilinear model for the mouse retina dataset.

### Structure:
#### Data Processing:
rna_seq_data_processing/: data_preprocessing_bc.ipynb and data_preprocessing_rgc.ipynb perform preprocessings of the bipolar cell (BC) and retinal ganglion cell (RGC) single-cell RNA-seq data. Follow procedures from this repo (https://github.com/shekharlab/RetinaEvolution) to get the input h5ad files.

eyewire_data_processing/: connection_matrix_proprocessing.ipynb processes Eyewire data for connectivity matrix, and connection_matrix_proprocessing_plot.ipynb generates relevant plots. Input data are sourced from Eyewire (https://museum.eyewire.org/) and are organized into pre_synaptic_cells.csv and post_synaptic_cells.csv.

#### Analysis:
figures/:
- bilinear_model.py contains the source code of the bilinear model (scenario 2) for the mouse retina dataset.
- bilinear_model_cv.ipynb performs 5-fold cross-validation, saving results in the folder named "cv_results".
- bilinear_model_final_training.ipynb performs the final training with the optimal hyperparameters selected from the cross-validation.
- bilinear_model_gene_analysis.ipynb performs the gene analysis given the transformation matrices from the final training.
- bilinear_model_plot.ipynb generates figures in the paper.
- bilinear_model_pred.ipynb generates predictions of the BC partners of transcriptionally-defined RGC types.

GO_analysis/: Gene Ontology (GO) enrichment analysis results.

## Requirements
Python 3.9.7
Additional Python libraries as required by individual applications (refer to the respective application for more details).

## Citation
Please cite the following manuscript if you use the bilinear model in your research:

Qiao, M. (2023+). Deciphering the Genetic Code of Neuronal Type Connectivity: A Bilinear Modeling Approach. Biorxiv. https://www.biorxiv.org/content/10.1101/2023.08.01.551532

## References:
1. Bae, JA., et al. (2018). Digital Museum of Retinal Ganglion Cells with Dense Anatomy and Physiology. Cell, 173(5):1293–1306.e19.
2. Kovacs IA., et al. (2020). Uncovering the genetic blueprint of the C. elegans nervous system. PNAS, 117(52):33570–33577.
3. Qiao, M. (2023+). Deciphering the Genetic Code of Neuronal Type Connectivity: A Bilinear Modeling Approach. Biorxiv.
4. Hahn, J., Monavarfeshani, A., Qiao, M., et al. (2023). Evolution of neuronal cell classes and types in the vertebrate retina. Nature, 624, 415–424.
