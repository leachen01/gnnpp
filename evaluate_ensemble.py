import argparse
import json
import numpy as np
import os
import pandas as pd
import pytorch_lightning as L
import torch
import torch_geometric

from dataclasses import dataclass
from models.graphensemble.multigraph import Multigraph
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from utils.data import (
    load_dataframes,
    load_distances,
    normalize_features_and_create_graphs,
    split_graph,
    rm_edges,
    summary_statistics,
)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("data", type=str, default="rf", help='Data to use for testing, can be "rf" or "f"')
    args.add_argument(
        "leadtime", type=str, default="24h", help='Leadtime to use for testing, can be "24h", "72h" or "120h"'
    )
    args.add_argument("folder", type=str, default="trained_models/best_24h", help="Folder to load the models from")

    args = args.parse_args()
    print("#################################################")
    print(f"[INFO] Starting evaluation with data: {args.data} and leadtime: {args.leadtime}")
    print("#################################################")

    CHECKPOINT_FOLDER = args.folder
    JSONPATH = os.path.join(CHECKPOINT_FOLDER, "params.json")

    # Load the JSON file
    with open(JSONPATH, "r") as f:
        print(f"[INFO] Loading {JSONPATH}")
        args_dict = json.load(f)

    @dataclass
    class DummyConfig:
        pass

    for key, value in args_dict.items():
        setattr(DummyConfig, key, value)

    config = DummyConfig()
    print("[INFO] Starting eval with config: ", args_dict)

    # Load Data ######################################################################
    dataframes = load_dataframes(mode="eval", leadtime=args.leadtime)
    # Only Summary ###################################################################
    only_summary = False
    if hasattr(config, "only_summary"):
        if config.only_summary is True or config.only_summary == "True":
            print("[INFO] Only using summary statistics...")
            dataframes = summary_statistics(dataframes)
            only_summary = True

    dist = load_distances(dataframes["stations"])
    graphs_train_rf, tests = normalize_features_and_create_graphs(
        training_data=dataframes["train"],
        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
        mat=dist,
        max_dist=config.max_dist,
    )
    graphs_test_rf, graphs_test_f = tests

    graphs_test = graphs_test_rf if args.data == "rf" else graphs_test_f

    if args.data == "f" and not only_summary:
        print("[INFO] Splitting graphs for f data...")
        graphs_split = [split_graph(g) for g in graphs_test]
        graphs_test = [g for sublist in graphs_split for g in sublist]

    # Remove Edges ##################################################################
    if hasattr(config, "remove_edges"):
        if config.remove_edges == "True" or config.remove_edges is True:
            print("[INFO] Removing edges...")
            rm_edges(graphs_train_rf)
            rm_edges(graphs_test)

    # Create Data Loaders ###########################################################
    print("[INFO] Creating data loaders...")
    train_loader = DataLoader(graphs_train_rf, batch_size=config.batch_size, shuffle=True)
    # test_loader_rf = DataLoader(graphs_test_rf, batch_size=1, shuffle=False)
    test_loader = DataLoader(graphs_test, batch_size=1 if args.data == "rf" else 5, shuffle=False)

    # Create Model ##################################################################
    print("[INFO] Creating ensemble...")

    emb_dim = 20
    in_channels = 55  # graphs_train_rf[0].x.shape[1] + emb_dim - 1

    FOLDER = os.path.join(CHECKPOINT_FOLDER, "models")
    preds_list = []
    for path in os.listdir(FOLDER):
        if path.endswith(".ckpt"):
            print(f"[INFO] Loading model from {path}")
            # Load Model from chekcpoint
            checkpoint = torch.load(os.path.join(FOLDER, path))

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

            multigraph.load_state_dict(checkpoint["state_dict"])

            trainer = L.Trainer(log_every_n_steps=1, accelerator="gpu", devices=1, enable_progress_bar=True)

            preds = trainer.predict(model=multigraph, dataloaders=[test_loader])

            if args.data == "f" and not only_summary:
                preds = [
                    prediction.reshape(5, 122, 2).mean(axis=0) for prediction in preds
                ]  # Average over the batch dimension

            preds = torch.cat(preds, dim=0)
            preds_list.append(preds)

    # ! Hacky wack of getting the targets
    targets = dataframes["test_rf"][1] if args.data == "rf" else dataframes["test_f"][1]
    targets = torch.tensor(targets.t2m.values) - 273.15

    stacked = torch.stack(preds_list)
    final_preds = torch.mean(stacked, dim=0)

    res = multigraph.loss_fn.crps(final_preds, targets)
    print("#############################################")
    print("#############################################")
    print(f"final crps: {res.item()}")
    print("#############################################")
    print("#############################################")

    # Save Results ##################################################################
    # Create DataFrame
    df = pd.DataFrame(np.concatenate([targets.view(-1, 1), final_preds], axis=1), columns=["t2m", "mu", "sigma"])
    df.to_csv(os.path.join(CHECKPOINT_FOLDER, f"{args.data}_results.csv"), index=False)

    # Create Log File ###############################################################
    log_file = os.path.join(CHECKPOINT_FOLDER, f"{args.data}.txt")
    with open(log_file, "w") as f:
        f.write(f"Data: {args.data}\n")
        f.write(f"Leadtime: {args.leadtime}\n")
        f.write(f"Final crps: {res.item()}")
