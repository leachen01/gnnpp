import os
import torch
from exploration.graph_creation import *
from utils.drn_utils import *
from torch.optim import AdamW
from models.graphensemble.multigraph import *
from models.drn import *

# get drn data, get graphs, get_json_save_result_path, get gnn and drn models

def get_drn_data(leadtime: str):
    # no early stopping, just as benchmark
    dataframes = load_dataframes(leadtime=leadtime)
    dataframes = summary_statistics(dataframes)
    dataframes.pop("stations")

    # test
    for X, y in dataframes.values():  # wofuer?
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

    train, valid_test = normalize_features(training_data=dataframes["train"], valid_test_data=[dataframes["test_rf"], dataframes["test_f"]])
    train = drop_nans(train)
    (test_rf, test_f) = valid_test
    test_rf = drop_nans(test_rf)
    test_f = drop_nans(test_f)
    return train, test_rf, test_f


def get_train_valid_graph_data(leadtime: str, graph_name: str):
    allowed_graphs = ["g1", "g2", "g3", "g4", "g5"]
    assert graph_name in allowed_graphs, f"{graph_name} is not in the list of allowed values"

    if graph_name == "g1":
        train_path = f'exploration/graphs/{leadtime}/{graph_name}_train_{leadtime}-AgEg50.pt'
        valid_path = f'exploration/graphs/{leadtime}/{graph_name}_valid_{leadtime}-AgEg50.pt'
        test_rf_path = f'exploration/graphs/{leadtime}/{graph_name}_test_rf_{leadtime}-AgEg50.pt'
        test_f_path = f'exploration/graphs/{leadtime}/{graph_name}_test_f_{leadtime}-AAgEg50.pt'

    if graph_name == "g2":
        train_path = f'exploration/graphs/{leadtime}/{graph_name}_train_{leadtime}-AgalloEg50.pt'
        valid_path = f'exploration/graphs/{leadtime}/{graph_name}_valid_{leadtime}-AgalloEg50.pt'
        test_rf_path = f'exploration/graphs/{leadtime}/{graph_name}_test_rf_{leadtime}-AgalloEg50.pt'
        test_f_path = f'exploration/graphs/{leadtime}/{graph_name}_test_f_{leadtime}-AgalloEg50.pt'

    if graph_name == "g3":
        train_path = f'exploration/graphs/{leadtime}/{graph_name}_train_{leadtime}-AgalloEg50-a4-o15.pt'
        valid_path = f'exploration/graphs/{leadtime}/{graph_name}_valid_{leadtime}-AgalloEg50-a4-o15.pt'
        test_rf_path = f'exploration/graphs/{leadtime}/{graph_name}_test_rf_{leadtime}-AgalloEg50-a4-o15.pt'
        test_f_path = f'exploration/graphs/{leadtime}/{graph_name}_test_f_{leadtime}-AgalloEg50-a4-o15.pt'

    if graph_name == "g4":
        train_path = f'exploration/graphs/{leadtime}/{graph_name}_train_{leadtime}-Ad2d3Ed2003-d30074.pt'
        valid_path = f'exploration/graphs/{leadtime}/{graph_name}_valid_{leadtime}-Ad2d3Ed2003-d30074.pt'
        test_rf_path = f'exploration/graphs/{leadtime}/{graph_name}_test_rf_{leadtime}-Ad2d3Ed2003-d30074.pt'
        test_f_path = f'exploration/graphs/{leadtime}/{graph_name}_test_f_{leadtime}-Ad2d3Ed2003-d30074.pt'

    if graph_name == "g5":
        train_path = f'exploration/graphs/{leadtime}/{graph_name}_train_{leadtime}-Agallod2d3Eg50-a4-o15-d2003-d30074.pt'
        valid_path = f'exploration/graphs/{leadtime}/{graph_name}_valid_{leadtime}-Agallod2d3Eg50-a4-o15-d2003-d30074.pt'
        test_rf_path = f'exploration/graphs/{leadtime}/{graph_name}_test_rf_{leadtime}-Agallod2d3Eg50-a4-o15-d2003-d30074.pt'
        test_f_path = f'exploration/graphs/{leadtime}/{graph_name}_test_f_{leadtime}-Agallod2d3Eg50-a4-o15-d2003-d30074.pt'

    path_list = [train_path, valid_path, test_rf_path, test_f_path]
    if os.path.exists(train_path) and os.path.exists(valid_path):
        print("Loading precomputed graph data...")
        try:
            train_data = torch.load(train_path)
            valid_data = torch.load(valid_path)
            test_rf = torch.load(test_rf_path)
            test_f = torch.load(test_f_path)
            print("Successfully loaded precomputed data.")
            return train_data, valid_data, test_rf, test_f
        except Exception as e:
            print(f"Error loading precomputed data: {e}")
            print("Falling back to data preparation...")
    else:
        print("Precomputed data not found.")

    print("Preparing data from scratch...")
    train_data, valid_data, test_rf, test_f = prepare_graph_data(leadtime=leadtime, graph_name=graph_name,
                                                           path_list=path_list)

    return train_data, valid_data, test_rf, test_f


def prepare_graph_data(leadtime: str, graph_name: str, path_list: list):

    dataframes = load_dataframes(leadtime=leadtime)
    dataframes = summary_statistics(dataframes)
    if graph_name == "g1":
        graphs_train_rf, tests = normalize_features_and_create_graphs1(df_train=dataframes['train'],
                                                                         df_valid_test=[dataframes['valid'],
                                                                                        dataframes['test_rf'],
                                                                                        dataframes['test_f']],
                                                                         station_df=dataframes['stations'],
                                                                         attributes=["geo"], edges=[("geo", 50)],
                                                                         sum_stats=True)

    if graph_name == "g2":
        graphs_train_rf, tests = normalize_features_and_create_graphs1(df_train=dataframes['train'],
                                                                         df_valid_test=[dataframes['valid'],
                                                                                        dataframes['test_rf'],
                                                                                        dataframes['test_f']],
                                                                         station_df=dataframes['stations'],
                                                                         attributes=["geo", "alt", "lon", "lat",
                                                                                     "alt-orog"], edges=[("geo", 50)],
                                                                         sum_stats=True)

    if graph_name == "g3":
        graphs_train_rf, tests = normalize_features_and_create_graphs1(df_train=dataframes['train'],
                                                                         df_valid_test=[dataframes['valid'],
                                                                                        dataframes['test_rf'],
                                                                                        dataframes['test_f']],
                                                                         station_df=dataframes['stations'],
                                                                         attributes=["geo", "alt", "lon", "lat",
                                                                                     "alt-orog"],
                                                                         edges=[("geo", 50), ("alt", 4), ("alt-orog", 1.5)],
                                                                         sum_stats=True)
    if graph_name == "g4":
        graphs_train_rf, tests = normalize_features_and_create_graphs1(df_train=dataframes['train'],
                                                                         df_valid_test=[dataframes['valid'],
                                                                                        dataframes['test_rf'],
                                                                                        dataframes['test_f']],
                                                                         station_df=dataframes['stations'],
                                                                         attributes=["dist2", "dist3"],
                                                                         edges=[("dist2", 0.003), ("dist3", 0.0074)],
                                                                         sum_stats=True)
    if graph_name == "g5":
        graphs_train_rf, tests = normalize_features_and_create_graphs1(df_train=dataframes['train'],
                                                                         df_valid_test=[dataframes['valid'],
                                                                                        dataframes['test_rf'],
                                                                                        dataframes['test_f']],
                                                                         station_df=dataframes['stations'],
                                                                         attributes=["geo", "alt", "lon", "lat",
                                                                                     "alt-orog", "dist2", "dist3"],
                                                                         edges=[("geo", 50), ("alt", 4),
                                                                                ("alt-orog", 1.5), ("dist2", 0.003),
                                                                                ("dist3", 0.0074)],
                                                                         sum_stats=True)
    graphs_valid_rf, graphs_test_rf, graphs_test_f = tests
    os.makedirs(f'exploration/graphs/{leadtime}', exist_ok=True)
    torch.save(graphs_train_rf, path_list[0])
    torch.save(graphs_valid_rf, path_list[1])
    torch.save(graphs_test_rf, path_list[2])
    torch.save(graphs_test_f, path_list[3])

    return graphs_train_rf, graphs_valid_rf, graphs_test_rf, graphs_test_f

def get_json_save_result_paths(leadtime: str, graph_name: str="g1", drn: bool=False):
    DIRECTORY = '/home/ltchen/gnnpp'
    if drn == True:
        SAVEPATH = os.path.join(DIRECTORY, f"leas_trained_models/drn_{leadtime}/models")
        JSONPATH = os.path.join(DIRECTORY, f"leas_trained_models/drn_{leadtime}/params.json")
        RESULTPATH = os.path.join(DIRECTORY, f"leas_trained_models/drn_{leadtime}")
    else:
        SAVEPATH = os.path.join(DIRECTORY, f"leas_trained_models/sum_stats_{leadtime}/{graph_name}_{leadtime}/models")
        JSONPATH = os.path.join(DIRECTORY, f"leas_trained_models/sum_stats_{leadtime}/{graph_name}_{leadtime}/params.json")
        RESULTPATH = os.path.join(DIRECTORY, f"leas_trained_models/sum_stats_{leadtime}/{graph_name}_{leadtime}/")
    return JSONPATH, SAVEPATH, RESULTPATH

def load_gnn_model(path: str, num_nodes, emb_dim, edge_dim, in_channels, config):
    multigraph_model = Multigraph.load_from_checkpoint(
        path,
        num_nodes=num_nodes,
        embedding_dim=emb_dim,
        edge_dim=edge_dim,
        in_channels=in_channels,
        hidden_channels_gnn=config['gnn_hidden'],
        out_channels_gnn=config['gnn_hidden'],
        num_layers_gnn=config['gnn_layers'],
        heads=config['heads'],
        hidden_channels_deepset=config['gnn_hidden'],
        optimizer_class=AdamW,
        optimizer_params=dict(lr=config['lr']),
        )
    multigraph_model.eval()
    return multigraph_model

def load_drn_model(path: str, embed_dim, in_channels, hidden_channels, lr):
    drn_model = DRN.load_from_checkpoint(
                path,
                embedding_dim=embed_dim,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                optimizer_class=AdamW,
                optimizer_params=dict(lr=lr),
            )
    drn_model.eval()
    return drn_model