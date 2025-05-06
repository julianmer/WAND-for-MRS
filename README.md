# WAND: WAVELET ANALYSIS-BASED NEURAL DECOMPOSITION OF MRS SIGNALS FOR ARTIFACT REMOVAL

[WAND: Wavelet Analysis-based Neural Decomposition of MRS Signals for Artifact Removal](https://doi.org/10.1002/nbm.70038)

## Abstract

Accurate quantification of metabolites in magnetic resonance spectroscopy (MRS) is challenged by low
signal-to-noise ratio (SNR), overlapping metabolites, and various artifacts. Particularly, unknown and
unparameterized baseline effects obscure the quantification of low-concentration metabolites, limiting MRS
reliability. This paper introduces wavelet analysis-based neural decomposition (WAND), a novel data-driven
method designed to decompose MRS signals into their constituent components: metabolite-specific signals,
baseline, and artifacts. WAND takes advantage of the enhanced separability of these components within
the wavelet domain. The method employs a neural network, specifically a U-Net architecture, trained to
predict masks for wavelet coefficients obtained through the continuous wavelet transform. These masks
effectively isolate desired signal components in the wavelet domain, which are then inverse-transformed
to obtain separated signals. Notably, an artifact mask is created by inverting the sum of all known signal
masks, enabling WAND to capture and remove even unpredictable artifacts. The effectiveness of WAND in
achieving accurate decomposition is demonstrated through numerical evaluations using simulated spectra.
Furthermore, WAND’s artifact removal capabilities significantly enhance the quantification accuracy of
linear combination model fitting. The method’s robustness is further validated using data from the 2016
MRS Fitting Challenge and in-vivo experiments.

## Overview

This repository consists of the following Python scripts:
* The `intro.ipynb` provides an introduction to the MRS data simulation and the WAND approach.
* The `train.py` implements the pipeline to train (and test) the deep learning approaches.
* The `sweep.py` defines ranges to sweep for optimal hyperparamters using Weights & Biases.
* The `frameworks/` folder holds the frameworks for model-based and data-driven methods.
  * The `framework.py` defines the framework class to inherit from.
  * The `frameworkFSL.py` consists of a wrapper for FLS-MRS for easy use.
  * The `frameworkLCM.py` is a wrapper for LCModel.
  * The `frameworkWAND.py` holds the framework class for the proposed WAND.
  * The `LCModel.exe` is the LCModel executable.
  * The `nnModels.py` defines the neural architectures.
* The `loading/` folder holds the scripts to automate the loading of MRS data formats.
  * The `dicom.py` defines functions for the DICOM loader (Philips).
  * The `lcmodel.py` contains loaders for the LCModel formats.
  * The `loadBasis.py` holds the loader for numerous basis set file formats.
  * The `loadConc.py` enables loading of concentration files provided by fitting software.
  * The `loadData.py` defines the loader for the MRS data.
  * The `philips.py` holds functions to load Philips data.
* The `simulation/` folder holds the scripts to simulate MRS spectra.
  * The `basis.py` has the basis set class to hold the spectra.
  * The `dataModules.py` are creating datasets by loading in-vivo data or simulating ad-hoc during training.
  * The `sigModels.py` defines signal models to simulate MRS spectra.
  * The `simulation.py` draws simulation parameters from distibutions to allow simulation with the signal model.
  * The `simulationDefs.py` holds predefined simulation parameters ranges.
* The `utils/` folder holds helpful functionalities.
  * The `auxiliary.py` defines some helpful functions.
  * The `components.py` consists of functions to create signal components.
  * The `gpu_config.py` is used for the GPU configuration.
  * The `processing.py` defines functions for processing MRS data.
  * The `wavelets.py` implements a differentiable CWT and ICWT.
* The `visualization/` folder holds scripts to visualize the results and other aspects.
  * The `plotFunctions.py` defines functions to plot the spectra, scalograms, results, etc.
  * The `visChallenge.py` visualizes the challenge data results.
  * The `visDecomposition.py` visualizes the decomposition results.
  * The `visInVivo.py` visualizes the in-vivo data results.
  * The `visInVivoLipids.py` visualizes the in-vivo lipid contamination data results.
  * The `visSynthetic.py` visualizes the synthetic data results.


## Requirements

| Module            | Version |
|:------------------|:-------:|
| fsl_mrs           | 2.1.17  |
| h5py              | 3.8.0   |
| matplotlib        | 3.7.0   |
| numpy             | 1.24.2  |
| pandas            | 1.5.3   |
| pydicom           | 2.3.1   |
| pytorch_lightning | 1.9.2   |
| scipy             | 1.10.0  |
| seaborn	    | 0.13.2  |
| shutup            | 0.2.0   |
| spec2nii          | 0.6.8   |
| ssqueezepy        | 0.6.4   |
| torch             | 1.13.1  |
| torchmetrics      | 0.11.1  |
| tqdm              | 4.64.1  |
| wandb             | 0.13.10 |
