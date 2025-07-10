import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import wandb

from models.drn import DRN
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from utils.data import load_dataframes, summary_statistics


def normalize_features(
    training_data: pd.DataFrame,
    valid_test_data: List[Tuple[pd.DataFrame]],
):
    # Normalize Features ############################################################
    # Select the features to normalize
    print("[INFO] Normalizing features...")
    train_rf = training_data[0]
    features_to_normalize = [col for col in train_rf.columns if col not in ["station_id", "time", "number"]]

    # Create a MinMaxScaler object
    scaler = StandardScaler()

    # Fit and transform the selected features
    train_rf.loc[:, features_to_normalize] = scaler.fit_transform(train_rf[features_to_normalize]).astype("float32")

    train_rf.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * train_rf["time"].dt.dayofyear / 365)
    train_rf.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * train_rf["time"].dt.dayofyear / 365)
    train_rf.drop(columns=["time", "number"], inplace=True)

    for features, _ in valid_test_data:
        features.loc[:, features_to_normalize] = scaler.transform(features[features_to_normalize]).astype("float32")
        features.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * features["time"].dt.dayofyear / 365)
        features.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * features["time"].dt.dayofyear / 365)
        features.drop(columns=["time", "number"], inplace=True)

    return training_data, valid_test_data


def drop_nans(dfs: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nans = dfs[1]["t2m"].isna().reset_index(drop=True)
    res = (dfs[0][~nans], dfs[1][~nans])
    return res


if __name__ == "__main__":
    with wandb.init(tags=["drn"], settings=wandb.Settings(_service_wait=300)):
        config = wandb.config
        # Load Data ####
        dataframes = load_dataframes(mode="hyperopt", leadtime=config.leadtime)
        dataframes = summary_statistics(dataframes)
        dataframes.pop("stations")

        for X, y in dataframes.values():
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)
        ################

        train, valid_test = normalize_features(training_data=dataframes["train"], valid_test_data=[dataframes["valid"]])
        valid = valid_test[0]

        # Drop Nans ####
        train = drop_nans(train)
        valid = drop_nans(valid)

        y_scaler = StandardScaler(with_std=False)
        y_scaler = y_scaler.fit(train[1][["t2m"]])

        train_dataset = TensorDataset(
            torch.Tensor(train[0].to_numpy()), torch.Tensor(y_scaler.transform(train[1][["t2m"]]))
        )
        valid_dataset = TensorDataset(
            torch.Tensor(valid[0].to_numpy()), torch.Tensor(y_scaler.transform(valid[1][["t2m"]]))
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)

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
        early_stop = EarlyStopping(monitor="val_loss", patience=10)
        trainer = L.Trainer(
            max_epochs=1000,
            log_every_n_steps=1,
            accelerator="gpu",
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=wandb_logger,
            callbacks=early_stop,
        )
        trainer.fit(model=drn, train_dataloaders=train_loader, val_dataloaders=valid_loader)
