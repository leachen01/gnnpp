import pytorch_lightning as L
import torch_geometric
import wandb

from models.graphensemble.multigraph import Multigraph
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from utils.data import (
    load_dataframes_old,
    load_distances,
    normalize_features_and_create_graphs,
    rm_edges,
    summary_statistics,
)


if __name__ == "__main__":
    with wandb.init():
        config = wandb.config
        print("[INFO] Starting sweep with config: ", config)

        # Load Data #####################################################################
        dataframes = load_dataframes_old(mode="hyperopt", leadtime=config.leadtime)
        if hasattr(config, "only_summary"):
            if config.only_summary is True or config.only_summary == "True":
                print("[INFO] Only using summary statistics...")
                dataframes = summary_statistics(dataframes)

        dist = load_distances(dataframes["stations"])
        graphs_train_rf, valid = normalize_features_and_create_graphs(
            training_data=dataframes["train"],
            valid_test_data=[dataframes["valid"]],
            mat=dist,
            max_dist=config.max_dist,
        )
        graphs_valid_rf = valid[0]
        # Remve edges for testing purposes ##############################################
        if hasattr(config, "remove_edges"):
            if config.remove_edges is True:
                print("[INFO] Removing edges...")
                rm_edges(graphs_train_rf)
                rm_edges(graphs_valid_rf)

        # Create Data Loaders ###########################################################
        print("[INFO] Creating data loaders...")
        train_loader = DataLoader(graphs_train_rf, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(graphs_valid_rf, batch_size=config.batch_size, shuffle=False)

        # Create Model ##################################################################
        print("[INFO] Creating model...")
        emb_dim = 20
        in_channels = graphs_train_rf[0].x.shape[1] + emb_dim - 1

        multigraph = Multigraph(
            embedding_dim=emb_dim,
            in_channels=in_channels,
            hidden_channels_gnn=config.gnn_hidden,
            out_channels_gnn=config.gnn_hidden,
            num_layers_gnn=config.gnn_layers,
            heads=config.heads,
            hidden_channels_deepset=config.gnn_hidden,
            optimizer_class=AdamW,
            optimizer_params=dict(lr=config.lr),
        )
        torch_geometric.compile(multigraph)

        # run a dummy forward pass to initialize the model
        multigraph.initialize(train_loader)

        wandb_logger = WandbLogger(project="multigraph")
        early_stop = EarlyStopping(monitor="val_loss", patience=10)

        # Train Model ###################################################################
        print("[INFO] Training model...")
        trainer = L.Trainer(
            max_epochs=1000,
            log_every_n_steps=1,
            accelerator="gpu",
            enable_progress_bar=True,
            logger=wandb_logger,
            callbacks=early_stop,
        )

        trainer.fit(model=multigraph, train_dataloaders=train_loader, val_dataloaders=valid_loader)
