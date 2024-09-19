from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import numpy as np
import pandas as pd


def normalize_features(
    training_data: pd.DataFrame,
    valid_test_data: List[Tuple[pd.DataFrame]],
) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame]]]:
    """
    Normalize the features in the training data and validation/test data. Also add the cos_doy and sin_doy features.

    Args:
        training_data (pd.DataFrame): The training data. Each Tuple contains the features and the target.
        valid_test_data (List[Tuple[pd.DataFrame]]): The validation/test data.
        Each Tuple contains the features and the target.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[pd.DataFrame]]]: The normalized training data and validation/test data.
    """

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
    """
    Drop rows with NaN values in the 't2m' column in the target Dataframe.

    Args:
        dfs (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing two DataFrames (Features and Target).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames with rows containing
            NaN values in the 't2m' column dropped.
    """
    nans = dfs[1]["t2m"].isna().reset_index(drop=True)
    res = (dfs[0][~nans], dfs[1][~nans])
    return res
