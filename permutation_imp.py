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
    shuffle_features,
)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("data", type=str, default="rf", help='Data to use for testing, can be "rf" or "f"')
    args.add_argument(
        "leadtime", type=str, default="24h", help='Leadtime to use for testing, can be "24h", "72h" or "120h"'
    )
    args.add_argument("feature_idx", type=int, default=0, help="Feature index to shuffle can range from 0 to 34")
    args.add_argument("model_idx", type=int, default=0, help="Index of model used for evaluation can range from 0 to 9")

    args = args.parse_args()
    print("#################################################")
    print(
        f"[INFO] Starting evaluation with data: {args.data} and leadtime: {args.leadtime}\
            \nUsing model {args.model_idx} and shuffling feature {args.feature_idx}"
    )
    print("#################################################")

    CHECKPOINT_FOLDER = f"trained_models/best_{args.leadtime}"
    JSONPATH = os.path.join(CHECKPOINT_FOLDER, "params.json")
    FOLDER = os.path.join(CHECKPOINT_FOLDER, "models")
    FEATURES = [
        "station_id",
        "cape",
        "model_orography",
        "sd",
        "station_altitude",
        "station_latitude",
        "station_longitude",
        "stl1",
        "swvl1",
        "t2m",
        "tcc",
        "tcw",
        "tcwv",
        "u10",
        "u100",
        "v10",
        "v100",
        "vis",
        "cp6",
        "mn2t6",
        "mx2t6",
        "p10fg6",
        "slhf6",
        "sshf6",
        "ssr6",
        "ssrd6",
        "str6",
        "strd6",
        "tp6",
        "z",
        "q",
        "u",
        "v",
        "t",
        "transformed_time",
        "no_permutation",
    ]

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

    preds_list = []
    # Load Data from disk ##############################################################
    dataframes = load_dataframes(mode="eval", leadtime=args.leadtime)
    dist = load_distances(dataframes["stations"])
    # ! Hacky wack of getting the targets
    targets = dataframes["test_rf"][1] if args.data == "rf" else dataframes["test_f"][1]
    targets = torch.tensor(targets.t2m.values) - 273.15

    # Create Graphs ######################################################################
    graphs_train_rf, tests = normalize_features_and_create_graphs(
        training_data=dataframes["train"],
        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
        mat=dist,
        max_dist=config.max_dist,
    )
    graphs_test_rf, graphs_test_f = tests
    graphs_test = graphs_test_rf if args.data == "rf" else graphs_test_f

    # Permutation Importance ########################################################
    # Shuffle the freautes in the test set to get a permutation and assess the impact on the CRPS
    feature_idx = [args.feature_idx]
    if args.feature_idx == 34:
        feature_idx = [34, 35]  # Time is a special case since it is sin and cos of time
    if args.feature_idx == 35:
        feature_idx = []  # Dont shuffle at all
    print(f"[INFO] Shuffling feature {FEATURES[args.feature_idx]} - {args.feature_idx}...")

    xs = [g.x for g in graphs_test]
    xs = torch.stack(xs)
    xs_shuffled = shuffle_features(xs=xs, feature_permute_idx=feature_idx)
    graphs_test_shuffled = []
    for graph, shuffled_feature in zip(graphs_test, xs_shuffled):
        graph.x = shuffled_feature
        graphs_test_shuffled.append(graph)

    # Split Graphs ######################################################################
    if args.data == "f":
        print("[INFO] Splitting graphs for f data...")
        graphs_split = [split_graph(g) for g in graphs_test_shuffled]
        graphs_test_shuffled = [g for sublist in graphs_split for g in sublist]

    # Create Data Loaders ###########################################################
    print("[INFO] Creating data loaders...")
    train_loader = DataLoader(graphs_train_rf, batch_size=config.batch_size, shuffle=True)
    bs = 1 if args.data == "rf" else 5
    test_loader = DataLoader(graphs_test_shuffled, batch_size=bs, shuffle=False)

    # Select the model to evaluate #####################################################
    models = os.listdir(FOLDER)
    models = [x for x in models if ".ckpt" in x]
    models.sort()
    path = models[args.model_idx]

    # Load Models ####################################################################
    print(f"[INFO] Loading model from {path}")
    emb_dim = 20
    in_channels = 55  # graphs_train_rf[0].x.shape[1] + emb_dim - 1

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

    # Load Model from chekcpoint
    checkpoint = torch.load(os.path.join(FOLDER, path))
    multigraph.load_state_dict(checkpoint["state_dict"])

    trainer = L.Trainer(log_every_n_steps=1, accelerator="gpu", devices=1, enable_progress_bar=True)

    preds = trainer.predict(model=multigraph, dataloaders=[test_loader])

    if args.data == "f":
        preds = [prediction.reshape(5, 122, 2).mean(axis=0) for prediction in preds]  # Average over the batch dimension

    print(f"\033[91m len(preds):{len(preds)} \033[0m")
    preds = torch.cat(preds, dim=0)
    print(f"\033[91m Preds.shape:{preds.shape} \033[0m")
    print(f"\033[91m Targets.shape:{targets.shape} \033[0m")
    res = multigraph.loss_fn.crps(preds, targets)
    score = res.item()
    # Save Results ##################################################################
    if not os.path.exists(os.path.join(CHECKPOINT_FOLDER, f"permutation_importance_{args.data}.csv")):
        imp = pd.DataFrame(np.zeros(shape=(10, len(FEATURES))) * np.nan, columns=FEATURES)
    else:
        imp = pd.read_csv(os.path.join(CHECKPOINT_FOLDER, f"permutation_importance_{args.data}.csv"))
    imp.iloc[args.model_idx, args.feature_idx] = score  # j is model number, i is feature number
    # Save as CSV
    imp.to_csv(os.path.join(CHECKPOINT_FOLDER, f"permutation_importance_{args.data}.csv"), index=False)
