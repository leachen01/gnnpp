# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import torch_geometric.datasets as datasets
# import torch_geometric.data as data
# import torch_geometric.transforms as transforms
# import networkx as nx
# from torch_geometric.utils import to_networkx
from scipy.interpolate import interp1d
from utils.data import *
from torch_geometric.utils import is_undirected, degree, contains_isolated_nodes
from tqdm import tqdm

def signed_difference(x, y): # macht es Sinn signed difference zu benutzen?
    return x - y

def dist_km(lat1: float = 0, lon1: float = 0, lat2: float = 0, lon2: float = 0) -> float:
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km

def signed_geodesic_km(lat1: float = 0, lon1: float = 0, lat2: float = 0, lon2: float = 0) -> float:
    if lat1 > lat2 or lon1 > lon2:
        dist = geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    else:
        dist = -1 * geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    return dist

def create_emp_cdf(station_temps):  # F_i(x)
    data_sorted = np.sort(station_temps)
    cdf = np.arange(len(data_sorted)) / len(data_sorted)
    cdf_function = interp1d(data_sorted, cdf, kind='previous', bounds_error=False, fill_value=(0, 1))
    return cdf_function

def dist2(i_id, j_id, train_set, sum_stats):
# def dist2(i_id, j_id):
    # print(i_id, j_id)
    t2m = 't2m'
    if sum_stats:
        t2m = 't2m_mean'
    i_train_temps = train_set[train_set['station_id'] == i_id][t2m]
    j_train_temps = train_set[train_set['station_id'] == j_id][t2m]
    F_i = create_emp_cdf(i_train_temps)
    F_j = create_emp_cdf(j_train_temps)
    sum = 0
    S = np.arange(train_set[t2m].min(), train_set[t2m].max(), 1)
    for x in S:
        sum += abs(F_i(x) - F_j(x))
    d2 = sum * 1/S.shape[0]
    return d2

def compute_d2_matrix(stations: pd.DataFrame, train_set: pd.DataFrame, sum_stats: bool) -> np.array: # nochmal checken ob die funktion noch funktionert!!
    station_id = np.array(stations.index).reshape(-1, 1)
    # print(station_id.shape)
    # print(station_id.T.shape)
    vectorized_dist2 = np.vectorize(dist2, excluded=[2])
    distance_matrix = vectorized_dist2(station_id, station_id.T, train_set, sum_stats)
    # distance_matrix = np.vectorize(dist2)(station_id, station_id.T)
    return distance_matrix

def load_d2_distances(stations: pd.DataFrame, train_set: pd.DataFrame, sum_stats: bool, leadtime: str) -> np.ndarray:
    if os.path.exists(f"/mnt/sda/Data2/gnnpp-data/d2_distances_EUPP_{leadtime}.npy"):
        print("[INFO] Loading distances from file...")
        mat = np.load(f"/mnt/sda/Data2/gnnpp-data/d2_distances_EUPP_{leadtime}.npy")
    else:
        print("[INFO] Computing distances...")
        mat = compute_d2_matrix(stations, train_set, sum_stats)
        np.save(f"/mnt/sda/Data2/gnnpp-data/d2_distances_EUPP_{leadtime}.npy", mat)
    return mat

def create_emp_cdf_of_errors(station_df, target_temp, sum_stats): # cdfs v
    t2m = 't2m'
    if sum_stats:
        t2m = 't2m_mean'
    f_bar = station_df.groupby(['time'])[t2m].mean()
    def cdf_functions(z):
        return (1/ station_df.nunique()['time']) * np.sum(f_bar.to_numpy() - target_temp.to_numpy() <= z)
    return cdf_functions

def dist3(i_id, j_id, cdfs):
    print(i_id, j_id)
    sum = 0
    S = np.arange(-10, 10, 0.5)
    for x in S:
        sum += abs(cdfs[i_id](x) - cdfs[j_id](x))
    d3 = sum * 1/S.shape[0]
    return d3

def compute_d3_matrix(stations: pd.DataFrame, train_set, train_target_set, sum_stats) -> np.array:
    station_id = np.array(stations.index).reshape(-1, 1)
    cdfs = []
    num_stations = len(train_set.station_id.unique())
    for i_id in range(0, num_stations):
        i_train = train_set[train_set['station_id'] == i_id]
        i_target_temps = train_target_set[train_target_set['station_id'] == i_id]['t2m']
        G_s = create_emp_cdf_of_errors(i_train, i_target_temps, sum_stats)
        cdfs.append(G_s)
    print("[INFO] Cdfs created.")
    vectorized_dist3 = np.vectorize(dist3, excluded=[2])
    distance_matrix = vectorized_dist3(station_id, station_id.T, cdfs)
    return distance_matrix

def load_d3_distances(stations: pd.DataFrame, train_set, train_target_set, sum_stats: bool, leadtime: str) -> np.ndarray:
    if os.path.exists(f"/mnt/sda/Data2/gnnpp-data/d3_distances_EUPP_{leadtime}.npy"):
        print("[INFO] Loading distances from file...")
        mat = np.load(f"/mnt/sda/Data2/gnnpp-data/d3_distances_EUPP_{leadtime}.npy")
    else:
        print("[INFO] Computing distances...")
        mat = compute_d3_matrix(stations, train_set, train_target_set, sum_stats)
        np.save(f"/mnt/sda/Data2/gnnpp-data/d3_distances_EUPP_{leadtime}.npy", mat)
    return mat

def load_d4_distances(stations: pd.DataFrame, train_set, train_target_set, sum_stats, leadtime: str) -> np.ndarray:
    mat_d2 = load_d2_distances(stations, train_set, sum_stats, leadtime=leadtime)
    mat_d3 = load_d3_distances(stations, train_set, train_target_set, sum_stats, leadtime=leadtime)
    mat = mat_d2 + mat_d3
    return mat

def compute_mat(station_df: pd.DataFrame, mode: str, sum_stats: bool = None, train_set: pd.DataFrame = None, train_target_set: pd.DataFrame = None, leadtime = None) -> np.array:
    if mode == "geo":
        lon = np.array(station_df["lon"].copy())
        lat = np.array(station_df["lat"].copy())
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        distance_matrix = np.vectorize(dist_km)(lat_mesh, lon_mesh, lat_mesh.T, lon_mesh.T)
    if mode == "alt":
        altitude = np.array(station_df["altitude"].copy())
        mesh1, mesh2 = np.meshgrid(altitude, altitude)
        distance_matrix = np.vectorize(signed_difference)(mesh1, mesh2) # zwei vektoren voneinander abziehen
    if mode == "alt-orog":
        altorog = np.array((station_df['altitude']-station_df['orog']).copy())
        mesh1, mesh2 = np.meshgrid(altorog, altorog)
        distance_matrix = np.vectorize(signed_difference)(mesh1, mesh2)
    if mode == "lon":
        lon = np.array(station_df["lon"].copy())
        mesh1, mesh2 = np.meshgrid(lon, lon)
        distance_matrix = np.vectorize(signed_geodesic_km)(lon1 =mesh1, lon2=mesh2)
    if mode == "lat":
        lat = np.array(station_df["lat"].copy())
        mesh1, mesh2 = np.meshgrid(lat, lat) # check if this meshgrid actually works!!
        distance_matrix = np.vectorize(signed_geodesic_km)(lat1=mesh1, lat2=mesh2) # vorzeichen!
    if mode == "dist2":
        distance_matrix = load_d2_distances(station_df, train_set, sum_stats, leadtime=leadtime)
    if mode == "dist3":
        distance_matrix = load_d3_distances(station_df, train_set, train_target_set, sum_stats = sum_stats, leadtime=leadtime)
    if mode == "dist4":
        distance_matrix = load_d4_distances(station_df, train_set, train_target_set, sum_stats=sum_stats, leadtime=leadtime)
    return distance_matrix

def get_adj(dist_matrix_sliced: np.array, max_dist: float = 50) -> np.array:
    mask = None
    mask = (dist_matrix_sliced <= max_dist) & (dist_matrix_sliced >= (-max_dist))
    diagonal = np.full((mask.shape[0], mask.shape[1]), True, dtype=bool)
    np.fill_diagonal(diagonal, False)
    mask = np.logical_and(mask, diagonal)
    return mask

def create_graph_data(
        df_train: Tuple[pd.DataFrame],
        date: str,
        ensemble: int = None,
        sum_stats: bool = False):
    day = df_train[0][df_train[0].time == date]
    if sum_stats:
        ens = day #?
    else:
        ens = day[day.number == ensemble] # only if sum_stats = False

    ens = ens.drop(columns=["time", "number"])
    x = torch.tensor(ens.to_numpy(dtype=np.float32))
    df_target = df_train[1]

    target = df_target[df_target.time == date]
    target = target.drop(columns=["time", "station_id"]).to_numpy(dtype=np.float32) - 273.15
    y = torch.tensor(target)
    # y = torch.tensor(target)
    lon = ens["station_longitude"].to_numpy().reshape(-1, 1)
    # print(lon.shape)
    lat = ens["station_latitude"].to_numpy().reshape(-1, 1)
    # print(lat.shape)
    position = np.concatenate([lon, lat], axis=1).reshape(-1, 2)
    # print(position.shape)
    # pos_dict = dict(enumerate(position))

    return x, y.squeeze(-1), position

def create_graph_dataset(
        df_train: pd.DataFrame,
        df_target: pd.DataFrame,
        station_df: pd.DataFrame,
        attributes: list,
        edges: list,
        ensemble: int = None,
        sum_stats: bool = False,
        leadtime: str = None,):
    assert (not ((ensemble == None) and (sum_stats == False))), "Input either ensemble member number or sum_stats=True"

    # assert all elements in edges exist in attributes!
    first_el = [t[0] for t in edges]
    assert set(attributes).issuperset(set(first_el)), "Edges must be created based on attributes that exist."

    # attribute tensor creation
    t_dim = len(attributes)
    num_stations = len(df_train.station_id.unique())
    attr_tensor = torch.empty((num_stations, num_stations, t_dim), dtype=torch.float32)
    for i, list_element in enumerate(attributes):
        # compute distance matrix
        attr_tensor[:,:,i] = torch.tensor(compute_mat(station_df=station_df, mode=list_element, sum_stats=sum_stats, leadtime=leadtime))

    attr_mask = torch.empty(num_stations, num_stations, len(edges))
    for i, el in enumerate(edges):
        attr, max_value = el
        # position von attr in der attribute liste => welche distance matrix in tensor
        pos = attributes.index(attr)
        attr_mask[:,:,i] = get_adj(attr_tensor[:, :, pos], max_dist=max_value)

    g_adj = attr_mask.any(dim=2)
    g_edges = np.array(np.argwhere(g_adj))
    g_edge_idx = torch.tensor(g_edges.T, dtype=torch.long)
    g_edge_attr = attr_tensor[g_adj]

    # standardization
    max_edge_attr = g_edge_attr.max(dim=0).values
    std_g_edge_attr = g_edge_attr / max_edge_attr

    n_nodes = len(df_train.station_id.unique())
    n_fc = len(df_train.number.unique())

    graphs = []
    for time in tqdm(df_train.time.unique()):
        x, y, position = create_graph_data((df_train, df_target), time, ensemble, sum_stats) # date raus!
        # graph = Data(x=x, edge_index=g_edge_idx.T, edge_attr=std_g_edge_attr, timestamp=time, y=y, pos=position, n_idx=torch.arange(n_nodes).repeat(n_fc))
        graph = Data(x=x, edge_index=g_edge_idx.T, edge_attr=std_g_edge_attr, timestamp=time, y=y,
                     n_idx=torch.arange(n_nodes).repeat(n_fc))
        graphs.append(graph)

    return graphs

def normalize_features(data: List[Tuple[pd.DataFrame]]):
    print("[INFO] Normalizing features...")
    train_rf = data[0][0]
    features_to_normalize = [col for col in train_rf.columns if col not in ["station_id", "time", "number"]]

    # Create a MinMaxScaler object
    scaler = StandardScaler()

    # Fit and transform the selected features
    for i, (features, targets) in enumerate(data):
        if i == 0:
            features.loc[:, features_to_normalize] = scaler.fit_transform(features[features_to_normalize]).astype("float32")
            print("fit_transform")
        else:

            features.loc[:, features_to_normalize] = scaler.transform(features[features_to_normalize]).astype("float32")
            print(f"transform {i}")
        features.loc[:, ["cos_doy"]] = np.cos(2 * np.pi * features["time"].dt.dayofyear / 365)
        features.loc[:, ["sin_doy"]] = np.sin(2 * np.pi * features["time"].dt.dayofyear / 365)
    return data


def create_one_graph(df_train: pd.DataFrame, df_target: pd.DataFrame, station_df: pd.DataFrame, attributes: list, edges: list, date: str, ensemble: int = None, sum_stats: bool = False, leadtime: str = None):
    '''
    FOR PLOTTING
    '''
    x, y, position = create_graph_data((df_train, df_target), date, ensemble, sum_stats)
    # assert all elements in edges exist in attributes!
    first_el = [t[0] for t in edges]
    assert set(attributes).issuperset(set(first_el)), "Edges must be created based on attributes that exist."

    # attribute tensor creation
    t_dim = len(attributes)
    num_stations = len(df_train.station_id.unique())
    attr_tensor = torch.empty((num_stations, num_stations, t_dim), dtype=torch.float32)
    for i, list_element in enumerate(attributes):
        # compute distance matrix
        attr_tensor[:,:,i] = torch.tensor(compute_mat(station_df, list_element, sum_stats, leadtime))

    attr_mask = torch.empty(num_stations, num_stations, len(edges))
    for i, el in enumerate(edges):
        attr, max_value = el
        # position von attr in der attribute liste => welche distance matrix in tensor
        pos = attributes.index(attr)
        attr_mask[:,:,i] = get_adj(attr_tensor[:, :, pos], max_dist=max_value)

    g_adj = attr_mask.any(dim=2)
    g_edges = np.array(np.argwhere(g_adj))
    g_edge_idx = torch.tensor(g_edges.T)
    g_edge_attr = attr_tensor[g_adj]

    # standardization
    max_edge_attr = g_edge_attr.max(dim=0).values
    std_g_edge_attr = g_edge_attr / max_edge_attr

    n_nodes = len(df_train.station_id.unique())
    n_fc = len(df_train.number.unique())

    graph = Data(x=x, edge_index=g_edge_idx.T, edge_attr=std_g_edge_attr, timestamp=date, y=y, pos=position, n_idx=torch.arange(n_nodes).repeat(n_fc))
    return graph

def normalize_features_and_create_graphs1(df_train: Tuple[pd.DataFrame], df_valid_test: List[Tuple[pd.DataFrame]], station_df: pd.DataFrame, attributes: list, edges: list, ensemble: int=None, sum_stats: bool = False, leadtime: str = None):

    list = [df_train] + df_valid_test
    dfs = normalize_features(list)
    # dfs = temp_conversion(dfs)
    test_valid = []

    for i, (features, targets) in enumerate(dfs):
        if i == 0:
            graphs_train_rf = create_graph_dataset(df_train=features, df_target=targets, station_df=station_df, attributes=attributes, edges=edges, ensemble = ensemble, sum_stats=sum_stats, leadtime=leadtime)

        else:
            graphs_valid_test = create_graph_dataset(df_train=features, df_target=targets, station_df=station_df, attributes=attributes, edges=edges, ensemble = ensemble, sum_stats=sum_stats, leadtime=leadtime)
            test_valid.append(graphs_valid_test)
    return graphs_train_rf, test_valid

def facts_about(graph):
    n_nodes = graph.num_nodes
    n_edges = graph.num_edges
    node_degrees = degree(graph.edge_index[0], num_nodes=n_nodes)
    avg_degree = node_degrees.mean().item()
    n_isolated_nodes = (node_degrees == 0).sum().item()
    feature_dim = graph.x.size(1)
    edge_dim = graph.num_edge_features

    print(f"Number of nodes: {n_nodes} with feature dimension of x: {feature_dim}")
    print(f"Number of isolated nodes: {n_isolated_nodes}")
    print(f"Number of edges: {n_edges} with edge dimension: {edge_dim}")
    print(f"Average node degree: {avg_degree}")