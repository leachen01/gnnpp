import argparse
import json
import os
import pytorch_lightning as L
import torch_geometric
import wandb

from models.graphensemble.multigraph import Multigraph
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from utils.data import (
    load_dataframes,
    load_distances,
    normalize_features_and_create_graphs,
    rm_edges,
    summary_statistics,
)

if __name__ == "__main__":

    if not os.path.exists("data/dataframes/final_train"):
        os.makedirs("data/dataframes/final_train")

    # Argparser to find right directory
    argparser = argparse.ArgumentParser()
    argparser.add_argument("leadtime", type=str, default="24h")
    argparser.add_argument("directory", type=str)
    argparser.add_argument("id", type=str)
    args = argparser.parse_args()

    DIRECTORY = args.directory
    JSONPATH = os.path.join(DIRECTORY, "params.json")
    SAVEPATH = os.path.join(DIRECTORY, "models")

    # Load the JSON file
    with open(JSONPATH, "r") as f:
        print(f"[INFO] Loading {JSONPATH}")
        args_dict = json.load(f)

    print(f"[INFO] Starting sweep with config: {args_dict} and id: {args.id}")
    with wandb.init(
        project="multigraph", id=f"training_run_{args.leadtime}_{args.id}", config=args_dict, tags=["final_training"]
    ):
        config = wandb.config
        print("[INFO] Starting sweep with config: ", config)

        # Load Data ######################################################################
        dataframes = load_dataframes(mode="train", leadtime=args.leadtime)
        dist = load_distances(dataframes["stations"])
        # Only Summary ###################################################################
        if hasattr(config, "only_summary"):
            if config.only_summary is True or config.only_summary == "True":
                print("[INFO] Only using summary statistics...")
                dataframes = summary_statistics(dataframes)
        # Normalize Features and Create Graphs ###########################################
        graphs_train_rf, tests = normalize_features_and_create_graphs(
            training_data=dataframes["train"],
            valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
            mat=dist,
            max_dist=config.max_dist,
        )
        graphs_test_rf, graphs_test_f = tests
        # Remove edges if specified #####################################################
        if hasattr(config, "remove_edges"):
            if config.remove_edges == "True" or config.remove_edges is True:
                print("[INFO] Removing edges...")
                rm_edges(graphs_train_rf)

        # Create Data Loaders ###########################################################
        print("[INFO] Creating data loaders...")
        train_loader = DataLoader(graphs_train_rf, batch_size=config.batch_size, shuffle=True)

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
        batch = next(iter(train_loader))
        batch = batch  # .to("cuda")
        multigraph  # .to("cuda")
        multigraph.forward(batch)

        wandb_logger = WandbLogger(project="multigraph")
        checkpoint_callback = ModelCheckpoint(
            dirpath=SAVEPATH, filename=f"run_{args.id}", monitor="train_loss", mode="min", save_top_k=1
        )

        # Train Model ###################################################################
        print("[INFO] Training model...")
        trainer = L.Trainer(
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
            logger=wandb_logger,
            callbacks=checkpoint_callback,
        )

        trainer.fit(model=multigraph, train_dataloaders=train_loader)

    wandb.finish()
