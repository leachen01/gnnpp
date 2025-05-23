import pytorch_lightning as L
import torch
import torch_geometric
import json
import sys
import os
import wandb

from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch.optim import AdamW
from pytorch_lightning.loggers import WandbLogger

project_root = '/home/ltchen/gnnpp'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("Added to path:", project_root)
# from models.loss import NormalCRPS
# from models.model_utils import MakePositive, EmbedStations
from utils.data import *
# from exploration.graph_creation import *
from models.graphensemble.multigraph import *
from exploration.get_graphs_and_data import *

# def get_train_valid_data(leadtime: str, graph_name:str):
#     train_path = f'exploration/graphs/{leadtime}/train_{leadtime}-Ad2d3Ed2003-d30074.pt'
#     valid_path = f'exploration/graphs/{leadtime}/valid_{leadtime}-Ad2d3Ed2003-d30074.pt'
#     test_rf_path = f'exploration/graphs/{leadtime}/test_rf_{leadtime}-Ad2d3Ed2003-d30074.pt'
#     test_f_path = f'exploration/graphs/{leadtime}/test_f_{leadtime}-Ad2d3Ed2003-d30074.pt'
#
#     path_list = [train_path, valid_path, test_rf_path, test_f_path]
#
#     if os.path.exists(train_path) and os.path.exists(valid_path):
#         print("Loading precomputed graph data...")
#         try:
#             train_data = torch.load(train_path)
#             valid_data = torch.load(valid_path)
#             test_rf = torch.load(test_rf_path)
#             test_f = torch.load(test_f_path)
#             print("Successfully loaded precomputed data.")
#             return train_data, valid_data, test_rf, test_f
#         except Exception as e:
#             print(f"Error loading precomputed data: {e}")
#             print("Falling back to data preparation...")
#     else:
#         print("Precomputed data not found.")
#
#     print("Preparing data from scratch...")
#     train_data, valid_data, test_rf, test_f = prepare_data(leadtime=leadtime, graph_name=graph_name, path_list=path_list)
#     return train_data, valid_data, test_rf, test_f
#
# def prepare_data(leadtime: str, graph_name:str, path_list: list):
#
#     dataframes = load_dataframes(leadtime=leadtime)
#     dataframes = summary_statistics(dataframes)
#     graphs4_train_rf, tests4 = normalize_features_and_create_graphs1(df_train=dataframes['train'],
#                                                                      df_valid_test=[dataframes['valid'], dataframes['test_rf'], dataframes['test_f']],
#                                                                      station_df=dataframes['stations'],
#                                                                      attributes=["dist2", "dist3"],
#                                                                      edges=[("dist2", 0.003), ("dist3", 0.0074)],
#                                                                      sum_stats=True)
#     graphs4_valid_rf, graphs4_test_rf, graphs4_test_f = tests4
#     os.makedirs(f'exploration/graphs/{leadtime}', exist_ok=True)
#     torch.save(graphs4_train_rf, path_list[0])
#     torch.save(graphs4_valid_rf, path_list[1])
#     torch.save(graphs4_test_rf, path_list[2])
#     torch.save(graphs4_test_f, path_list[3])
#     return graphs4_train_rf, graphs4_valid_rf, graphs4_test_rf, graphs4_test_f

if __name__ == "__main__":
    graph_name = "g4"
    leadtime = "120h"
    graphs4_train_rf, graphs4_valid_rf, graphs4_test_rf, graphs4_test_f  = get_train_valid_graph_data(leadtime=leadtime, graph_name=graph_name)

    g4_train_loader = DataLoader(graphs4_train_rf, batch_size=8, shuffle=True)
    g4_valid_loader = DataLoader(graphs4_valid_rf, batch_size=8, shuffle=False)
    g4_test_f_loader = DataLoader(graphs4_test_f, batch_size=8, shuffle=False)
    g4_test_rf_loader = DataLoader(graphs4_test_rf, batch_size=8, shuffle=False)

    train_loader = g4_train_loader
    valid_loader = g4_valid_loader
    # test_f_loader = g4_test_f_loader
    # test_rf_loader = g4_test_rf_loader
    # test_loader = [test_f_loader, test_rf_loader]

    emb_dim = 20
    in_channels = graphs4_train_rf[0].x.shape[1] + emb_dim - 1
    edge_dim = graphs4_train_rf[0].num_edge_features
    num_nodes = graphs4_train_rf[0].num_nodes
    PROJECTNAME = "sweep_g4" # wandb sweep --project sweep_g4 sweep_g4.yaml

    TRAINNAME = f"{graph_name}_{leadtime}_train_run"

    with wandb.init():
        config = wandb.config
        print("[INFO] Starting sweep with config: ", config)

        multigraph = Multigraph(
            num_nodes=num_nodes,
            embedding_dim=emb_dim,
            edge_dim=edge_dim,
            in_channels=in_channels,
            hidden_channels_gnn=config['gnn_hidden'],
            out_channels_gnn=config['gnn_hidden'],
            num_layers_gnn=config['gnn_layers'],
            heads=config['heads'],
            hidden_channels_deepset=config['gnn_hidden'],
            optimizer_class=AdamW,
            optimizer_params=dict(lr=config['lr']),
        )
        # batch = next(iter(train_loader))
        # multigraph.forward(batch)
        multigraph.initialize(train_loader)

        wandb_logger = WandbLogger(project=PROJECTNAME)
        early_stop = EarlyStopping(monitor="val_loss", patience=10)
        # progress_bar = TQDMProgressBar(refresh_rate=0)

        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=SAVEPATH, filename=TRAINNAME, monitor="val_loss", mode="min", save_top_k=1
        # )

        trainer = L.Trainer(
            max_epochs=1000,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=[1],
            enable_progress_bar=True,
            logger=wandb_logger,
            # callbacks=[early_stop, progress_bar, checkpoint_callback],
            callbacks=early_stop,
        )

        trainer.fit(model=multigraph, train_dataloaders=train_loader, val_dataloaders=valid_loader)