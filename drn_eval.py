import argparse
import json
import numpy as np
import os
import pandas as pd
import pytorch_lightning as L
import torch

from dataclasses import dataclass
from models.drn import DRN
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from utils.data import load_dataframes, summary_statistics
from utils.drn_utils import normalize_features, drop_nans

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
    dataframes = load_dataframes(mode="train", leadtime=config.leadtime)
    dataframes = summary_statistics(dataframes)
    dataframes.pop("stations")

    for X, y in dataframes.values():
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    ##################################################################################

    train, (test_rf, test_f) = normalize_features(
        training_data=dataframes["train"], valid_test_data=[dataframes["test_rf"], dataframes["test_f"]]
    )

    # Drop Nans ######################################################################
    train = drop_nans(train)
    test = test_rf if args.data == "rf" else test_f
    test = drop_nans(test)

    y_scaler = StandardScaler(with_std=False)
    y_scaler = y_scaler.fit(train[1][["t2m"]])

    test_dataset = TensorDataset(torch.Tensor(test[0].to_numpy()), torch.Tensor(y_scaler.transform(test[1][["t2m"]])))

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    embed_dim = 20
    in_channels = train[0].shape[1] + embed_dim - 1
    # Eval Ensemble ##################################################################
    FOLDER = os.path.join(CHECKPOINT_FOLDER, "models")
    preds_list = []
    drn = DRN(
        in_channels=in_channels,
        hidden_channels=config.hidden_channels,
        embedding_dim=embed_dim,
        optimizer_class=AdamW,
        optimizer_params=dict(lr=config.lr),
    )
    for path in os.listdir(FOLDER):
        if not path.endswith(".ckpt"):
            continue
        print(f"[INFO] Loading model from {path}")
        # Load Model from chekcpoint
        checkpoint = torch.load(os.path.join(FOLDER, path))

        drn.load_state_dict(checkpoint["state_dict"])

        trainer = L.Trainer(
            log_every_n_steps=1, accelerator="gpu", enable_progress_bar=True, enable_model_summary=False
        )

        preds = trainer.predict(model=drn, dataloaders=test_loader)
        preds = torch.cat(preds, dim=0)
        # Reverse transform of the y_scaler (only on the mean)
        preds[:, 0] = torch.Tensor(y_scaler.inverse_transform(preds[:, 0].view(-1, 1))).flatten()
        preds_list.append(preds)

    # Targets #######################################################################
    targets = test[1]
    targets = torch.Tensor(targets.t2m.values)

    stacked = torch.stack(preds_list)
    final_preds = torch.mean(stacked, dim=0)

    res = drn.loss_fn.crps(final_preds, targets)
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
