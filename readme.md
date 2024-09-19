# Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts

This repository contains the code of the paper Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts.

## Data

In this Study the EUPPBench dataset ([Demaeyer et al., 2023](https://essd.copernicus.org/articles/15/2635/2023/)) is used, which is publicly available on [Zenodo](https://zenodo.org/records/7708362) or using [climetlab-eumetnet-postprocessing-benchmark
](https://github.com/EUPP-benchmark/climetlab-eumetnet-postprocessing-benchmark).

## Files and Folders

### 🕸️ GNN Training and Evaluation

📄 sweep.py: Hyperparameter sweep for GNN models.  
📄 train_ensemble.py:  Train a GNN, given the parameters in trained_models/X_XXh/params.json.  
📄 evaluate_ensemble.py: Load an ensemble of trained GNNs and evaluate their averaged prediction.  

📁 **Models**  
📄 drn.py: Reimplementatioin of DRN (Rasp & Lerch, 2018) in PyTorch  
📄 loss.py: CRPS loss functions  
📄 model_utils.py: Utility Functions for the Embedding or ensuring positivity of the predicted $\sigma$  
📄 benchmark_models.py: Implementation of different GNN architectures used in graphensemble/multigraph.py  

### 🌦️ DRN

📄 drn_sweep.py: Hyperparameter sweep for the DRN.  
📄 drn_train.py: Train a DRN, given the parameters in trained_models/drn_XXh/params.json.  
📄 drn_eval.py: Load an ensemble of trained DRNs and evaluate their averaged prediction.  

### 🛠️ Others

📁 Trained Models: Folder where restults of trained models are stored.
📁 Utils: Utility functions for plotting, data loading and preparation.  

### 📈 Further Results and Plotting

📄 evaluation_plots_etc.ipynb: Almost all plots in the paper are generated here.  
📄 further_results.ipynb: Used to calculate PI length and coverage.  
📄 permutation_imp.py: Calculate feature importance.  
📄 stations.ipynb: Plot of map of stations.  
