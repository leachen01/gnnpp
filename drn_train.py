import argparse
import json
import os
import pytorch_lightning as L
import torch
import wandb

from models.drn import DRN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from utils.data import load_dataframes, summary_statistics
from utils.drn_utils import normalize_features, drop_nans

if __name__ == "__main__":
    if not os.path.exists("data/dataframes/final_train"):
        os.makedirs("data/dataframes/final_train")

    # Argparser to find right directory
    argparser = argparse.ArgumentParser()
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

    print(f"[INFO] Starting training with config: {args_dict} and id: {args.id}")
    with wandb.init(
        project="multigraph",
        id=f"training_run_drn_{args_dict['leadtime']}_{args.id}",
        config=args_dict,
        tags=["final_training"],
    ):
        config = wandb.config

        dataframes = load_dataframes(mode="train", leadtime=config.leadtime)
        dataframes = summary_statistics(dataframes)
        dataframes.pop("stations")

        for X, y in dataframes.values():
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)
        ################

        train, valid_test = normalize_features(
            training_data=dataframes["train"], valid_test_data=[dataframes["test_rf"], dataframes["test_f"]]
        )

        # Drop Nans ####
        train = drop_nans(train)

        y_scaler = StandardScaler(with_std=False)
        y_scaler = y_scaler.fit(train[1][["t2m"]])

        train_dataset = TensorDataset(
            torch.Tensor(train[0].to_numpy()), torch.Tensor(y_scaler.transform(train[1][["t2m"]]))
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        embed_dim = 20
        in_channels = train[0].shape[1] + embed_dim - 1

        drn = DRN(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            embedding_dim=embed_dim,
            optimizer_class=AdamW,
            optimizer_params=dict(lr=config.lr),
        )
        wandb_logger = WandbLogger(project="multigraph")
        checkpoint_callback = ModelCheckpoint(
            dirpath=SAVEPATH, filename=f"run_{args.id}", monitor="train_loss", mode="min", save_top_k=1
        )
        trainer = L.Trainer(
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            accelerator="gpu",
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=wandb_logger,
            callbacks=checkpoint_callback,
        )
        trainer.fit(model=drn, train_dataloaders=train_loader)
