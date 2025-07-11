# Improving Graph Neural Networks for Ensemble Post-Processing in Weather Forecasting

This repository contains the code of the bachelor thesis 'Improving Graph Neural Networks for Ensemble Post-Processing in Weather Forecasting'.

## Data

Following [Feik et al., 2024](https://arxiv.org/abs/2407.11050), we use the EUPPBench dataset ([Demaeyer et al., 2023](https://essd.copernicus.org/articles/15/2635/2023/)) in this thesis, which is publicly available on [Zenodo](https://zenodo.org/records/7708362) or using [climetlab-eumetnet-postprocessing-benchmark
](https://github.com/EUPP-benchmark/climetlab-eumetnet-postprocessing-benchmark).

## Files and Folders

### Exploration
Contains the main implementation for GNN model runs and explainability.

**📁 analysis**: 
- eda120, eda122: EDA for 120 and 122 stations. The thesis will continue with 120 stations due to the missing labels.
- check_122vs120_difference: run both datasets and check results
- drn_run: Reproduction of DRN (Feik et al., 2024)
- pred_crps_values: CRPS evaluation for one run instead of model ensemble

**📁 explainability**: 

- gnnexplainer.ipynb: exploration of gnnexplainer, creation of MultigraphWrapper
- gnnexplainer_oneseed.ipynb: cleaner version of gnnexplainer.ipynb, accounting for randomness using seed 42 and averaging over multiple GNNExplainer explanations
- gnnexplainer_generate_figures.py: generate figures for thesis.
- permutation_study.ipynb: permutation importance for both DRN and all GNN models

**📁 gnn**:
- gnn_run4: running models for all lead times for reforecast and forecast

**📁 graph_creation_file**: 
- create_graph_dataset: functions created (are copied to utils.data now)
- visualisation_final: using functions to create graphs for plotting

**📁 plot**: 
- graph_eda: plot nan map and altitude map, plot distribution of distance metrics, plot dist2-dist3 relationship,
- plot_crps_improvement: CRPSS per station compared to DRN CRPS as reference, also comparing CRPS with $g_1$ as reference 
- plot_dist2dist3: plot empirical CDFs used for dist2 and dist3
- plot_eda: EDA for dataset, CRPS theoretical background plots
- plot_pi: permutation importance plots following Feik et al. (2024)
- plot_pit: plot PIT histograms following Feik et al. (2024)

**📁 sweeps**: 
- for hyperparameter tuning of all three lead times and corresponding graphs $g_3$, $g_4$, and $g_5$
 
📄 get_graphs_and_data: saving and retrieving saved graph data so that it does not need to be created every single time all again  
📄 gnn_train: GNN training taken from Feik et al. (2024), not used in this thesis  
📄 graph_creation: all functions needed to create a graph (distance metrics calculations, graph construction, etc.)

### Leas_trained_models
Contains the parameters for all configurations for both DRN and the ensemble summary statistics.  
📁 drn: Hyperparameters from [Feik et al., 2024](https://arxiv.org/abs/2407.11050) for lead times: 24h, 72h, 120h; forecast types: rf, f.    
📁 sum_stats: Using hyperparameters from hyperparameter sweep in exploration/sweeps

### Models
Contains the GNN and DRN model architecture from [Feik et al., 2024](https://arxiv.org/abs/2407.11050), they are adapted by the integration of edge attributes and the number of stations  
📁 Graphensemble: Folder with the GNN model architecture.  
📄 drn: DRN model architecture.    
📄 loss: Model loss calculation, ignores NaN values in loss calculation.    
📄 model_utils: Utils used for GNN models.  

### Utils
Contains all utility functions for EDA, DRN model, explainability, and plotting.  
📄 data.py: loading data for EDA in anaylsis/eda120, and creating graphs by computing distances and adjacency matrices.    
📄 data122.py: copy of data but without dropping stations 62 and 74, used for EDA in analysis/eda.    
📄 drn_utils.py: NaN values are dropped and features are normalized.  
📄 explainability_utils.py:   
- Permutation Importance: the whole feature list, shuffling of DRN features.  
- GNNExplainer: MultigraphWrapper, gnnexplainer creation for MultigraphWrapper  
📄 plot.py: For plotting cartopy map.  
