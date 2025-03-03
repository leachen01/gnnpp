import geopy.distance
import numpy as np
import os
import pandas as pd
import torch
import torch_geometric
import xarray as xr

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from typing import DefaultDict, Tuple, List, Union


class ZarrLoader:
    """
    A class for loading data from Zarr files.

    Args:
        data_path (str): The path to the data directory.

    Attributes:
        data_path (str): The path to the data directory.
        leadtime (pd.Timedelta): The lead time for the forecasts.
        features (List[str]): The list of features to load.

    Methods:
        get_stations(arr: xarray.Dataset) -> pd.DataFrame:
            Get the stations information from the dataset.

        load_data(leadtime: str = "24h",
        features: Union[str, List[str]] = "all")
        -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset, xarray.Dataset]:
            Load the data from Zarr files.

        validate_stations() -> bool:
            Validate if the station IDs match between forecasts and reforecasts.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def get_stations(self, arr: xr.Dataset) -> pd.DataFrame:
        """
        Get the stations information from the dataset.

        Args:
            arr (xarray.Dataset): The dataset containing station information.

        Returns:
            pd.DataFrame: The dataframe containing station information.
        """
        stations = pd.DataFrame(
            {
                "station_id": arr.station_id.values,
                "lat": arr.station_latitude.values,
                "lon": arr.station_longitude.values,
                "altitude": arr.station_altitude.values,
                "orog": arr.model_orography.values,
                "name": arr.station_name.values,
            }
        )
        stations = stations.sort_values("station_id").reset_index(drop=True)
        return stations

    def load_raw_data(self) ->  Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
        ZARRDATAFOLDER = '/mnt/sda/Data2/gnnpp-data/EUPPBench-stations/'
        print(f"[INFO] Loading training data (1997-2013)")
        xr_train = xr.open_zarr(f'{ZARRDATAFOLDER}train.zarr')
        xr_train_targets = xr.open_zarr(f'{ZARRDATAFOLDER}train_targets.zarr')

        # Forecasts
        print(f"[INFO] Loading data for forecasts (2017-2018)")
        xr_forecasts = xr.open_zarr(f'{ZARRDATAFOLDER}test_f.zarr')
        xr_targets_f = xr.open_zarr(f'{ZARRDATAFOLDER}test_f_targets.zarr')

        # Reforecasts
        print(f"[INFO] Loading data for reforecasts (2014-2017)")
        xr_reforecasts = xr.open_zarr(f'{ZARRDATAFOLDER}test_rf.zarr')
        xr_targets_rf = xr.open_zarr(f'{ZARRDATAFOLDER}test_rf_targets.zarr')

        df_train = (
            xr_train.to_dataframe()
            #.reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )

        print("df_train loaded")
        df_train_target = (
            xr_train_targets.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            #.reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )
        print("df_train_targets loaded")
        df_f = (
            xr_forecasts.to_dataframe()
            #.reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )
        print("df_f loaded")
        df_f_target = (
            xr_targets_f.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            #.reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )

        print(f"df_f_targets {df_f_target.columns}")

        df_rf = (
            xr_reforecasts.to_dataframe()
            #.reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )

        print(f"df_rf{df_rf.columns}")
        df_rf_target = (
            xr_targets_rf.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            #.reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )

        print("df_rf_targets loaded")

        station_ids = df_f.station_id.unique()
        id_to_index = {station_id: i for i, station_id in enumerate(station_ids)}

        df_train["station_id"] = df_train["station_id"].apply(lambda x: id_to_index[x])
        df_train_target["station_id"] = df_train_target["station_id"].apply(lambda x: id_to_index[x])
        df_f["station_id"] = df_f["station_id"].apply(lambda x: id_to_index[x])
        df_f_target["station_id"] = df_f_target["station_id"].apply(lambda x: id_to_index[x])
        df_rf["station_id"] = df_rf["station_id"].apply(lambda x: id_to_index[x])
        df_rf_target["station_id"] = df_rf_target["station_id"].apply(lambda x: id_to_index[x])

        return df_train, df_train_target, df_f, df_f_target, df_rf, df_rf_target


    def load_data(
            self, leadtime: str = "24h", features: Union[str, List[str]] = "all"
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
        self.leadtime = pd.Timedelta(leadtime)

        ZARRDATAFOLDER = '/mnt/sda/Data2/gnnpp-data/EUPPBench-stations/'

        if features == "all":
            print("[INFO] Loading all features")
            self.features = ["number"] + [
                "station_id",
                "time",
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
            ]
        elif isinstance(features, list):
            print(f"[INFO] Loading features: {features}")
            self.features = ["number"] + features
        else:
            raise ValueError("features must be a list of strings or 'all'")

        print(f"[INFO] Loading training data (1997-2013)")
        xr_train = xr.open_zarr(f'{ZARRDATAFOLDER}train.zarr')
        xr_train = xr_train.sel(step=leadtime).drop_vars(["step"])
        print(xr_train)
        xr_train_targets = xr.open_zarr(f'{ZARRDATAFOLDER}train_targets.zarr')
        xr_train_targets = xr_train_targets.sel(step=leadtime).drop_vars(["step"])

        # Forecasts
        print(f"[INFO] Loading data for forecasts (2017-2018)")
        xr_forecasts = xr.open_zarr(f'{ZARRDATAFOLDER}test_f.zarr')
        xr_forecasts = xr_forecasts.sel(step=leadtime).drop_vars(["step"])
        xr_targets_f = xr.open_zarr(f'{ZARRDATAFOLDER}test_f_targets.zarr')
        xr_targets_f = xr_targets_f.sel(step=leadtime).drop_vars(["step"])

        # Reforecasts
        print(f"[INFO] Loading data for reforecasts (2014-2017)")
        xr_reforecasts = xr.open_zarr(f'{ZARRDATAFOLDER}test_rf.zarr')
        xr_reforecasts = xr_reforecasts.squeeze(drop=True).sel(step=leadtime).drop_vars(["step"])
        xr_targets_rf = xr.open_zarr(f'{ZARRDATAFOLDER}test_rf_targets.zarr')
        xr_targets_rf = xr_targets_rf.squeeze(drop=True).sel(step=leadtime).drop_vars(["step"])

        # extract stations
        self.stations_train = self.get_stations(xr_train)
        self.stations_f = self.get_stations(xr_forecasts)
        self.stations_rf = self.get_stations(xr_reforecasts)

        xr_train = xr_train.drop_vars(
            ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage"]
        )
        print(xr_train)
        xr_forecasts = xr_forecasts.drop_vars(
            ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage"]
        )
        xr_reforecasts = xr_reforecasts.drop_vars(
            ["model_altitude", "model_land_usage", "model_latitude", "model_longitude", "station_land_usage"]
        )
        print(
            f"[INFO] Data loaded successfully. Forecasts shape:\
            {xr_forecasts.t2m.shape}, Reforecasts shape: {xr_reforecasts.t2m.shape}"
        )

        df_train = (
            xr_train.to_dataframe()
            .reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )

        print("df_train loaded")
        df_train_target = (
            xr_train_targets.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            .reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )
        print("df_train_targets loaded")
        df_f = (
            xr_forecasts.to_dataframe()
            .reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )
        print("df_f loaded")
        df_f_target = (
            xr_targets_f.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            .reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )

        print(f"df_f_targets {df_f_target.columns}")

        df_rf = (
            xr_reforecasts.to_dataframe()
            .reorder_levels(["time", "number", "station_id"])
            .sort_index(level=["time", "number", "station_id"])
            .reset_index()
        )

        print(f"df_rf{df_rf.columns}")
        df_rf_target = (
            xr_targets_rf.t2m.drop_vars(["altitude", "land_usage", "latitude", "longitude", "station_name"])
            .to_dataframe()
            .reorder_levels(["time", "station_id"])
            .sort_index(level=["time", "station_id"])
            .reset_index()
        )

        print("df_rf_targets loaded")

        station_ids = df_f.station_id.unique()
        id_to_index = {station_id: i for i, station_id in enumerate(station_ids)}

        df_train["station_id"] = df_train["station_id"].apply(lambda x: id_to_index[x])
        df_train_target["station_id"] = df_train_target["station_id"].apply(lambda x: id_to_index[x])
        df_f["station_id"] = df_f["station_id"].apply(lambda x: id_to_index[x])
        df_f_target["station_id"] = df_f_target["station_id"].apply(lambda x: id_to_index[x])
        df_rf["station_id"] = df_rf["station_id"].apply(lambda x: id_to_index[x])
        df_rf_target["station_id"] = df_rf_target["station_id"].apply(lambda x: id_to_index[x])

        # Cut features #### But this puts it in the wrong order! => check again when used!!
        df_train = df_train[self.features]
        df_f = df_f[self.features]
        df_rf = df_rf[self.features]

        # df_train: (1997-2013), df_f: (2017-2018), df_rf: (2014-2017)

        return df_train, df_train_target, df_f, df_f_target, df_rf, df_rf_target

    def validate_stations(self):
        test1 = (self.stations_f.station_id == self.stations_rf.station_id).all()
        test2 = (self.stations_train.station_id == self.stations_rf.station_id).all()
        test3 = (self.stations_f.station_id == self.stations_train.station_id).all()
        return (test1 and test2 and test3)


def load_dataframes(
    mode: str,
    leadtime: str,
) -> DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load the dataframes for training, testing on reforecasts, and testing on forecasts either from Zarr
    or as a pandas Dataframe. If the dataframes do not exist as pandas Dataframe,
    they are created and saved to disk so future loading is faster.

    Args:
        mode (str): The mode of the script, can be "train", "eval", or "hyperopt".
        leadtime (str): The leadtime of the predictions, can be "24h", "72h", or "120h".

    Returns:
        DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the dataframes for
        training, validation and testing.

    """

    # Load Data ######################################################################
    DATA_FOLDER = f"/mnt/sda/Data2/gnnpp-data/orog_dataframes_{leadtime}" #orog change!
    res = defaultdict(lambda: None)

    if mode == "train" or mode == "eval":
        DATA_FOLDER = os.path.join(DATA_FOLDER, "final_train")
        # Training data
        TRAIN_RF_PATH = os.path.join(DATA_FOLDER, "train_rf_final.pkl")
        TRAIN_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "train_rf_target_final.pkl")
        # Test on Reforceasts
        TEST_RF_PATH = os.path.join(DATA_FOLDER, "valid_rf_final.pkl")
        TEST_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "valid_rf_target_final.pkl")
        # Test on Forecasts
        TEST_F_PATH = os.path.join(DATA_FOLDER, "test_f_final.pkl")
        TEST_F_TARGET_PATH = os.path.join(DATA_FOLDER, "test_f_target_final.pkl")

        STATIONS_PATH = os.path.join(DATA_FOLDER, "stations.pkl")

        # Check if the files exist
        if (
            os.path.exists(TRAIN_RF_PATH)
            and os.path.exists(TRAIN_RF_TARGET_PATH)
            and os.path.exists(TEST_RF_PATH)
            and os.path.exists(TEST_RF_TARGET_PATH)
            and os.path.exists(TEST_F_PATH)
            and os.path.exists(TEST_F_TARGET_PATH)
            and os.path.exists(STATIONS_PATH)
        ):

            print("[INFO] Dataframes exist. Will load pandas dataframes.")
            train_rf = pd.read_pickle(TRAIN_RF_PATH)
            train_rf_target = pd.read_pickle(TRAIN_RF_TARGET_PATH)

            test_rf = pd.read_pickle(TEST_RF_PATH)
            test_rf_target = pd.read_pickle(TEST_RF_TARGET_PATH)

            test_f = pd.read_pickle(TEST_F_PATH)
            test_f_target = pd.read_pickle(TEST_F_TARGET_PATH)

            stations_f = pd.read_pickle(STATIONS_PATH)

        else:
            print("[INFO] Data files not found, will load from zarr.")
            loader = ZarrLoader("mnt/sda/Data2/gnnpp-data/EUPPBench-stations")

            print("[INFO] Loading data...")
            df_train, df_train_target, df_f, df_f_target, df_rf, df_rf_target = loader.load_data(
                leadtime=leadtime, features="all"
            )
            assert loader.validate_stations(), "Stations in forecasts and reforecasts do not match."
            stations_f = loader.stations_f

            # Split the data
            # Test 2014-2017 # 4 years (Forecasts)
            # Test2 2014-15 # 2 years (Reforecasts)
            # Now train with full data
            # Train 1997-2013 # 13 years (Reforecasts)
            train_cutoff = pd.Timestamp("2014-01-01")
            train_rf = df_train.loc[df_train["time"] < train_cutoff, :]
            train_rf_target = df_train_target.loc[df_train_target["time"] < train_cutoff, :]

            test_rf = df_rf.loc[(df_rf["time"] >= train_cutoff), :]
            test_rf_target = df_rf_target.loc[(df_rf_target["time"] >= train_cutoff), :]

            test_f = df_f
            test_f_target = df_f_target

            if not os.path.exists(DATA_FOLDER):
                os.makedirs(DATA_FOLDER)
            print("[INFO] Saving dataframes to disk...")
            train_rf.to_pickle(TRAIN_RF_PATH)
            train_rf_target.to_pickle(TRAIN_RF_TARGET_PATH)

            test_rf.to_pickle(TEST_RF_PATH)
            test_rf_target.to_pickle(TEST_RF_TARGET_PATH)

            test_f.to_pickle(TEST_F_PATH)
            test_f_target.to_pickle(TEST_F_TARGET_PATH)

            stations_f.to_pickle(STATIONS_PATH)

        res["train"] = (train_rf, train_rf_target)
        res["test_rf"] = (test_rf, test_rf_target)
        res["test_f"] = (test_f, test_f_target)
        res["stations"] = stations_f
        return res

    if mode == "hyperopt":
        # Training data
        TRAIN_RF_PATH = os.path.join(DATA_FOLDER, "train_rf.pkl")
        TRAIN_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "train_rf_target.pkl")
        # Test on Reforecasts
        VALID_RF_PATH = os.path.join(DATA_FOLDER, "valid_rf.pkl")
        VALID_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "valid_rf_target.pkl")

        STATIONS_PATH = os.path.join(DATA_FOLDER, "stations.pkl")

        # Check if the files exist
        if (
            os.path.exists(TRAIN_RF_PATH)
            and os.path.exists(TRAIN_RF_TARGET_PATH)
            and os.path.exists(VALID_RF_PATH)
            and os.path.exists(VALID_RF_TARGET_PATH)
            and os.path.exists(STATIONS_PATH)
        ):
            print("[INFO] Dataframes exist. Will load pandas dataframes.")
            train_rf = pd.read_pickle(TRAIN_RF_PATH)
            train_rf_target = pd.read_pickle(TRAIN_RF_TARGET_PATH)
            valid_rf = pd.read_pickle(VALID_RF_PATH)
            valid_rf_target = pd.read_pickle(VALID_RF_TARGET_PATH)
            stations_f = pd.read_pickle(STATIONS_PATH)

        else:
            print("[INFO] Data files not found, will load from zarr.")
            loader = ZarrLoader("mnt/sda/Data2/gnnpp-data/EUPPBench-stations")
            print("[INFO] Loading data...")
            df_train, df_train_target, df_f, df_f_target, df_rf, df_rf_target = loader.load_data(
                leadtime=leadtime, features="all"
            )
            assert loader.validate_stations(), "Stations in forecasts and reforecasts do not match."
            stations_f = loader.stations_f
            # Split the data
            # df_train: (1997-2013), df_f: (2017-2018), df_rf: (2014-2017)
            # Test 2014-2017 # 4 years (forecasts)
            # Test2 2014-15 # 2 years (reforecasts) # !OTHER PAPER USED 4 YEARS (2012-2015)
            # Valid 2010-2013 # 4 years
            # Train 1997-2009 # 13 years

            train_cutoff = pd.Timestamp("2010-01-01")
            valid_cutoff = pd.Timestamp("2014-01-01")

            train_rf = df_train.loc[df_train["time"] < train_cutoff, :]
            train_rf_target = df_train_target.loc[df_train_target["time"] < train_cutoff, :]

            valid_rf = df_train.loc[(df_train["time"] >= train_cutoff) & (df_train["time"] < valid_cutoff), :]
            valid_rf_target = df_train_target.loc[
                (df_train_target["time"] >= train_cutoff) & (df_train_target["time"] < valid_cutoff), :
            ]

            if not os.path.exists(DATA_FOLDER):
                os.makedirs(DATA_FOLDER)
            print("[INFO] Saving dataframes to disk...")
            train_rf.to_pickle(TRAIN_RF_PATH)
            train_rf_target.to_pickle(TRAIN_RF_TARGET_PATH)
            valid_rf.to_pickle(VALID_RF_PATH)
            valid_rf_target.to_pickle(VALID_RF_TARGET_PATH)
            stations_f.to_pickle(STATIONS_PATH)

        res["train"] = (train_rf, train_rf_target)
        res["valid"] = (valid_rf, valid_rf_target)
        res["stations"] = stations_f

        return res

def load_complete_dataframe() -> DefaultDict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    DATA_FOLDER = f"/mnt/sda/Data2/gnnpp-data/dataframes_complete"
    res = defaultdict(lambda: None)

    DATA_FOLDER = os.path.join(DATA_FOLDER, "final_train")
    # Training data
    TRAIN_RF_PATH = os.path.join(DATA_FOLDER, "train_rf_final.pkl")
    TRAIN_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "train_rf_target_final.pkl")
    # Test on Reforecasts
    TEST_RF_PATH = os.path.join(DATA_FOLDER, "valid_rf_final.pkl")
    TEST_RF_TARGET_PATH = os.path.join(DATA_FOLDER, "valid_rf_target_final.pkl")
    # Test on Forecasts
    TEST_F_PATH = os.path.join(DATA_FOLDER, "test_f_final.pkl")
    TEST_F_TARGET_PATH = os.path.join(DATA_FOLDER, "test_f_target_final.pkl")

    if (
        os.path.exists(TRAIN_RF_PATH)
        and os.path.exists(TRAIN_RF_TARGET_PATH)
        and os.path.exists(TEST_RF_PATH)
        and os.path.exists(TEST_RF_TARGET_PATH)
        and os.path.exists(TEST_F_PATH)
        and os.path.exists(TEST_F_TARGET_PATH)
    ):

        print("[INFO] Dataframes exist. Will load pandas dataframes.")
        train_rf = pd.read_pickle(TRAIN_RF_PATH)
        train_rf_target = pd.read_pickle(TRAIN_RF_TARGET_PATH)

        test_rf = pd.read_pickle(TEST_RF_PATH)
        test_rf_target = pd.read_pickle(TEST_RF_TARGET_PATH)

        test_f = pd.read_pickle(TEST_F_PATH)
        test_f_target = pd.read_pickle(TEST_F_TARGET_PATH)

    else:
        print("[INFO] Data files not found, will load from zarr.")
        loader = ZarrLoader("mnt/sda/Data2/gnnpp-data/EUPPBench-stations")

        print("[INFO] Loading data...")
        df_train, df_train_target, df_f, df_f_target, df_rf, df_rf_target = loader.load_raw_data()

        # Split the data
        # Test 2014-2017 # 4 years (Forecasts)
        # Test2 2014-15 # 2 years (Reforecasts)
        # Now train with full data
        # Train 1997-2013 # 13 years (Reforecasts)
        train_rf = df_train
        train_rf_target = df_train_target

        test_rf = df_rf
        test_rf_target = df_rf_target

        test_f = df_f
        test_f_target = df_f_target

        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        print("[INFO] Saving dataframes to disk...")
        train_rf.to_pickle(TRAIN_RF_PATH)
        train_rf_target.to_pickle(TRAIN_RF_TARGET_PATH)

        test_rf.to_pickle(TEST_RF_PATH)
        test_rf_target.to_pickle(TEST_RF_TARGET_PATH)

        test_f.to_pickle(TEST_F_PATH)
        test_f_target.to_pickle(TEST_F_TARGET_PATH)

    res["train"] = (train_rf, train_rf_target)
    res["test_rf"] = (test_rf, test_rf_target)
    res["test_f"] = (test_f, test_f_target)
    return res

def load_stations(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create a DataFrame containing station-specific data from the input DataFrame.

    :param df: The DataFrame created by load_data.
    :type df: pd.DataFrame

    Returns:
        Tuple[pd.DataFrame, np.ndarray]]: Dataframe with stations and numpy array with station positions
    """
    stations = df.groupby(by="station")[["lat", "lon", "alt", "orog"]].first().reset_index()
    stations.station = pd.to_numeric(stations.station, downcast="integer")

    postions_matrix = np.array(stations[["station", "lon", "lat"]])
    return stations, postions_matrix


def load_distances(stations: pd.DataFrame) -> np.ndarray:
    """Load the distance matrix from file if it exists, otherwise compute it and save it to file.

    Args:
        stations (pd.DataFrame): The stations dataframe.

    Returns:
        np.ndarray: The distance matrix.
    """
    # Load Distances #################################################################
    if os.path.exists("data/distances_EUPP.npy"):
        print("[INFO] Loading distances from file...")
        mat = np.load("/mnt/sda/Data2/gnnpp-data/distances_EUPP.npy")
    else:
        print("[INFO] Computing distances...")
        mat = compute_dist_matrix(stations)
        np.save("/mnt/sda/Data2/gnnpp-data/distances_EUPP.npy", mat)
    return mat


def dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the distance between two stations in kilometers using the WGS-84 ellipsoid.

    :param lat1: Latitude of the first station.
    :type lat1: float
    :param lat2: Latitude of the second station.
    :type lat2: float
    :param lon1: Longitude of the first station.
    :type lon1: float
    :param lon2: Longitude of the second station.
    :type lon2: float

    :return: The distance between the two stations in kilometers.
    :rtype: float
    """
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km


def compute_dist_matrix(df: pd.DataFrame) -> np.array:
    """
    Returns a distance matrix between stations.

    :param df: dataframe with stations

    :return: distance matrix
    :rtype: np.array
    """
    coords_df = df[["lat", "lon"]].copy()

    # create numpy arrays for latitudes and longitudes
    latitudes = np.array(coords_df["lat"])
    longitudes = np.array(coords_df["lon"])

    # create a meshgrid of latitudes and longitudes
    lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes)

    # calculate distance matrix using vectorized distance function
    distance_matrix = np.vectorize(dist_km)(lat_mesh, lon_mesh, lat_mesh.T, lon_mesh.T)
    return distance_matrix


def normalize_features_and_create_graphs(
    training_data: pd.DataFrame,
    valid_test_data: List[Tuple[pd.DataFrame]],
    mat: np.ndarray,
    max_dist: float,
) -> Tuple[List[torch_geometric.data.Data], List[List[torch_geometric.data.Data]]]:
    """
    Normalize the features in the training data and create graph data.

    Args:
        training_data (pd.DataFrame): The training data.
        valid_test_data (List[Tuple[pd.DataFrame]]): The validation and test data. Each Tuple consists of the Features
        and Targets.
        mat (np.ndarray): The distance matrix.
        max_dist (float): The maximum distance.

    Returns:
        Tuple[List[torch_geometric.data.Data], List[List[torch_geometric.data.Data]]]:
        A tuple containing the graph data for the training data and the validation/test data.
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

    for features, targets in valid_test_data:
        features.loc[:, features_to_normalize] = scaler.transform(features[features_to_normalize]).astype("float32")
        features.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * features["time"].dt.dayofyear / 365)
        features.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * features["time"].dt.dayofyear / 365)

    # Create Graph Data ##############################################################
    # ! a conversion from kelvin to celsius is also done in create_multigraph
    print("[INFO] Creating graph data...")
    graphs_train_rf = create_multigraph(df=train_rf, df_target=training_data[1], distances=mat, max_dist=max_dist)

    test_valid = []
    for features, targets in valid_test_data:
        graphs_valid_test = create_multigraph(df=features, df_target=targets, distances=mat, max_dist=max_dist)
        test_valid.append(graphs_valid_test)

    return graphs_train_rf, test_valid


def split_graph(graph) -> List[torch_geometric.data.Data]:
    """Splits a graph which is created using 51 ensemble members into 5 subgraphs,
    each containing 10 or 11 ensemble members.

    Args:
        graph (torch_geometric.data.Data): the graph to be split
    """
    perm = torch.randperm(51) * 122  # First node of each ensemble member
    index = perm[:, None] + torch.arange(122)  # Add the node indices to each ensemble member

    set1 = index[:10]
    set2 = index[10:20]
    set3 = index[20:30]
    set4 = index[30:40]
    set5 = index[40:]  # Has 11 elements

    sets = [
        set1,
        set2,
        set3,
        set4,
        set5,
    ]  # Each set contains a list of station indices corresponding to 10 (or 11) ensemble members
    graphs = []
    for s in sets:
        graphs.append(graph.subgraph(s.flatten()))
    return graphs


def shuffle_features(xs: torch.Tensor, feature_permute_idx: List[int]) -> torch.Tensor:
    """Shuffle a tensor of the shape [T, N, F] first along the T dimension and then along the N dimension

    Args:
        xs (torch.Tensor): [T, N, F]
        feature_permute_idx (List[int]): indices of the features to permute
        (can be used to permute certain features together)

    Returns:
        torch.tensor: the shuffled tensor
    """

    xs_permuted = xs[..., feature_permute_idx]  # [T, N, F]

    T, N, _ = xs_permuted.shape
    perm_T = torch.randperm(T)  # First permute the features in time
    xs_permuted = xs_permuted[perm_T, ...]

    # Then permute the features within each ensemble member
    # Shuffle across N dimension, but do so differently for each time step T
    indices = torch.argsort(torch.rand((T, N)), dim=1).unsqueeze(-1).repeat(1, 1, len(feature_permute_idx))
    result = torch.gather(xs_permuted, dim=1, index=indices)

    # Replace features with permuted features
    xs[..., feature_permute_idx] = result
    return xs


def rm_edges(data: List[torch_geometric.data.Data]) -> None:
    """Remove all edges from the graphs in the list.

    Args:
        data (List[torch_geometric.data.Data]): List of graphs
    """
    for graph in data:
        graph.edge_index = torch.empty(2, 0, dtype=torch.long)
        graph.edge_attr = torch.empty(0)


def summary_statistics(dataframes: defaultdict) -> defaultdict:
    """
    Calculate summary statistics for each feature dataframe in the given dictionary.
    The dictionary can contain multiple tuples which contain the dataframe and the target values.
    Also the dict can contain a dataframe with the stations, which will be returned unaltered.

    Args:
        dataframes (defaultdict): A dictionary containing the dataframes to calculate summary statistics for.

    Returns:
        defaultdict: A dictionary containing the updated dataframes with summary statistics.

    """
    only_mean = ["model_orography", "station_altitude", "station_latitude", "station_longitude"]
    for key, df in dataframes.items():
        if key == "stations":
            continue
        print(f"[INFO] Calculating summary statistics for {key}")
        y = df[1]
        df = df[0]

        rest = [col for col in df.columns if col not in only_mean]

        mean_agg = df.groupby(["time", "station_id"])[only_mean].agg("mean")
        rest_agg = (
            df.groupby(["time", "station_id"])[rest].agg(["mean", "std"]).drop(columns=["number", "station_id", "time"])
        )
        rest_agg.columns = ["_".join(col).strip() for col in rest_agg.columns.values]
        df = pd.concat([mean_agg, rest_agg], axis=1).reset_index()
        df["number"] = 0
        dataframes[key] = (df, y)
    return dataframes


def get_mask(
    dist_matrix_sliced: np.array, method: str = "max_dist", k: int = 3, max_dist: int = 50, nearest_k_mode: str = "in"
) -> np.array:
    """
    Generate mask which specifies which edges to include in the graph
    :param dist_matrix_sliced: distance matrix with only the reporting stations
    :param method: method to compute included edges. max_dist includes all edges wich are shorter than max_dist km,
    knn includes the k nearest edges for each station. So the out_degree of each station is k,
    the in_degree can vary.
    :param k: number of connections per node
    :param max_dist: maximum length of edges
    :param nearest_k_mode: "in" or "out". If "in" the every node has k nodes passing information to it,
    if "out" every node passes information to k nodes.
    :return: return boolean mask of which edges to include
    :rtype: np.array
    """
    mask = None
    if method == "max_dist":
        mask = (dist_matrix_sliced <= max_dist) & (dist_matrix_sliced != 0)
    elif method == "knn":
        k = k + 1
        nearest_indices = np.argsort(dist_matrix_sliced, axis=1)[:, 1:k]
        # Create an empty boolean array with the same shape as distances
        nearest_k = np.zeros_like(dist_matrix_sliced, dtype=bool)
        # Set the corresponding indices in the nearest_k array to True
        row_indices = np.arange(dist_matrix_sliced.shape[0])[:, np.newaxis]
        nearest_k[row_indices, nearest_indices] = True
        if nearest_k_mode == "in":
            mask = nearest_k.T
        elif nearest_k_mode == "out":
            mask = nearest_k

    return mask


def generate_layers(n_nodes, n_layers) -> np.array:
    """
    Generate bidirectional connections between nodes x_{s,n}, where s has the same value.

    Args:
        n_nodes (int): Number of stations (122).
        n_layers (int): Number of ensemble members (11 or 51).

    Returns:
        np.array: Bidirectional connections between layers of nodes.

    """
    all_layers = []
    start_i = 0
    start_j = n_nodes
    for i in range(n_layers):
        layer_i = np.arange(start_i, start_i + n_nodes)
        start_j = start_i + n_nodes
        for j in range(i + 1, n_layers):
            layer_array = np.empty((2, n_nodes), dtype=int)
            layer_array[0] = layer_i
            layer_array[1] = np.arange(start_j, start_j + n_nodes)
            all_layers.append(layer_array)
            start_j += n_nodes
        start_i += n_nodes
    connections = np.hstack(all_layers)
    connections_bidirectional = np.hstack([connections, np.flip(connections, axis=0)])
    return connections_bidirectional


def create_multigraph(df, df_target, distances, max_dist):
    """Create a multigraph from the input data.

    Args:
        df (pd.DataFrame)): feature dataframe
        df_target (pd.DataFrame): target dataframe
        distances (np.ndarray): distance matrix
        max_dist (float): maximum distance for edges

    Returns:
        List[torch_geometric.data.Data]: List of graphs with features and targets
    """
    n_nodes = len(df.station_id.unique())  # number of nodes
    n_fc = len(df.number.unique())  # number of ensembe members
    df = df.drop(columns=["number"])

    # Create set of edges ######################################################################
    mask = get_mask(distances, max_dist=max_dist)
    edges = np.argwhere(mask)
    edge_index = edges.T  # (2, num_edges) holds indices of connected nodes for one level (forecast) of the multigraph

    # Create edge features
    edge_attr = distances[edges[:, 0], edges[:, 1]]
    edge_attr = edge_attr.reshape(-1, 1)
    max_len = np.max(edge_attr)
    standardized_edge_attr = edge_attr / max_len

    # Repeat edge_attr for all levels of the multigraph
    full_edge_attr = np.repeat(standardized_edge_attr, n_fc, axis=1).T.reshape(-1, 1)

    # Get all other Levels of the Multigraph
    values_to_add = np.arange(n_fc) * (n_nodes)
    stacked = np.repeat(edge_index[np.newaxis, ...], n_fc, axis=0) + values_to_add[:, np.newaxis, np.newaxis]
    full_edge_index = stacked.transpose(1, 0, 2).reshape(2, -1)

    # Add connections between levels
    if n_fc > 1:
        full_edge_index = np.hstack([full_edge_index, generate_layers(n_nodes, n_layers=n_fc)])

    # Add 0 for each remaining edge attribute
    connections = (
        np.ones((full_edge_index.shape[1] - full_edge_attr.shape[0], 1)) * 0.01
    )  # fill connections with small value
    full_edge_attr = np.vstack([full_edge_attr, connections])
    full_edge_attr = torch.tensor(full_edge_attr, dtype=torch.float32)

    # Create node features ######################################################################
    graphs = []
    for time in df.time.unique():
        day = df[df.time == time]  # get all forecasts for one day
        day = day.drop(columns=["time"])
        x = torch.tensor(day.to_numpy(dtype=np.float32))
        assert x.shape[0] == n_nodes * n_fc

        day_target = df_target[df_target.time == time]
        # ! TODO This should surely be done somerwhere else
        y = torch.tensor(day_target["t2m"].to_numpy(dtype=np.float32)) - 273.15  # ! convert to celsius
        assert y.shape[0] == n_nodes, f"y.shape[0] = {y.shape[0]}, n_nodes = {n_nodes}, time = {time}, {day_target}"
        pyg_data = Data(
            x=x,
            edge_index=torch.tensor(full_edge_index, dtype=torch.long),
            edge_attr=full_edge_attr,
            y=y,
            timestamp=time,
            n_idx=torch.arange(n_nodes).repeat(n_fc),
        )
        graphs.append(pyg_data)
    return graphs
