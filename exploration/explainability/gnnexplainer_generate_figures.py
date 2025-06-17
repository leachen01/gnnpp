import sys
import json
project_root = '/home/ltchen/gnnpp'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("Added to path:", project_root)
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from torch_geometric.utils import to_networkx
from utils.explainability_utils import *
from utils.plot import plot_map
from exploration.get_graphs_and_data import *

#%%
DIRECTORY = os.getcwd()
print(DIRECTORY)


import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(graph_name):
    leadtime = "72h"
    # graph_name = "g1"

    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
    # DIRECTORY = os.getcwd()

    JSONPATH, SAVEPATH, RESULTPATH = get_json_save_result_paths(leadtime=leadtime, graph_name=graph_name)
    with open(JSONPATH, "r") as f:
        print(f"[INFO] Loading {JSONPATH}")
        args_dict = json.load(f)
    config = args_dict

    g_train_rf, g_valid_rf, g_test_rf, g_test_f = get_train_valid_graph_data(leadtime=leadtime, graph_name=graph_name)

    g1_train_loader = DataLoader(g_train_rf, batch_size=config['batch_size'], shuffle=True)
    g1_valid_loader = DataLoader(g_valid_rf, batch_size=config['batch_size'], shuffle=True)
    g1_test_f_loader = DataLoader(g_test_f, batch_size=config['batch_size'], shuffle=False)
    g1_test_rf_loader = DataLoader(g_test_rf, batch_size=config['batch_size'], shuffle=False)

    train_loader = g1_train_loader
    valid_loader = g1_valid_loader
    test_f_loader = g1_test_f_loader
    test_rf_loader = g1_test_rf_loader
    test_loader = [test_f_loader, test_rf_loader]

    emb_dim = 20
    in_channels = g_train_rf[0].x.shape[1] + emb_dim - 1
    edge_dim = g_train_rf[0].num_edge_features
    num_nodes = g_train_rf[0].num_nodes
    #%%
    dataframes = load_dataframes(leadtime)
    dataframes = summary_statistics(dataframes)

    for _, d in enumerate(['test_rf', 'test_f', 'train']):
    # d = "test_rf"
        TRAINNAME = f"{graph_name}_{leadtime}_train_run0"
        CKPT_PATH = os.path.join(SAVEPATH, TRAINNAME + '.ckpt')
        multigraph = load_gnn_model(CKPT_PATH, num_nodes, emb_dim, edge_dim, in_channels, config)

        if d == "train":
            graph_to_explain = g_train_rf[0]
        if d == "test_rf":
            graph_to_explain = g_test_rf[0]
        if d == "test_f":
            graph_to_explain = g_test_f[0]

        multigraph = multigraph.to('cpu')
        graph_to_explain = graph_to_explain.to('cpu')

        for _, node_idx in enumerate([5]):
            # node_idx = 5 # 23 #5: g1: lr=0.04, es=0.01; g5: lr=0.04, edge_size=0.2, g4: 1000 epochs: lr=0.01, edge_size=0.005
            print(f"############ Starting {node_idx} with {d}")
            FIGUREPATH = os.path.join(DIRECTORY, f'figures/results/gnnexplainer_new/{graph_name}/{node_idx}')
            os.makedirs(FIGUREPATH, exist_ok=True)
            if node_idx == 118:
                lr= 0.04
                edge_size=0.01
                if graph_name == "g1":
                    neighbors = [116, 117, 118, 119, 63, 49]
                else:
                    neighbors = [116, 117, 118, 119, 63]
            else:
                if graph_name=="g1":
                    lr = 0.04
                    edge_size = 0.01
                    neighbors = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11]
                elif graph_name=="g3":
                    lr = 0.04
                    edge_size = 0.2
                    neighbors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 23, 25, 75, 76, 33, 79, 85, 86]
                elif graph_name=="g4":
                    lr = 0.01
                    edge_size = 0.005
                    neighbors = [6, 10]
                else:
                    lr = 0.04
                    edge_size = 0.2
                    neighbors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 75, 76, 33, 79, 85, 86]

            degrees = degree(graph_to_explain.edge_index[0])
            num_neighbors = degrees[node_idx].item()
            print(f"Node {node_idx} has {num_neighbors} neighbors")

            df = pd.DataFrame(columns=['run_id', 'node', 'feature', 'weight'])
            weights_dict = {}
            num_runs = 5

            for run_id in range(num_runs):
                set_seed(42*run_id)
                mexplainer = create_explainer(multigraph, epochs=1000, lr=lr, edge_size=edge_size) # lr= 0.04, edge_size = 0.03
                explanation = mexplainer(x=graph_to_explain.x, edge_index=graph_to_explain.edge_index,
                                         edge_attr=graph_to_explain.edge_attr, index=node_idx)
                print(f'Generated explanations in {explanation.available_explanations}')

            # Feature importance ################################################################
                feature_loc = torch.nonzero(explanation.node_mask,
                                            as_tuple=False)  # feature_loc.shape = 384 (64*6), 2 (node, feature)
                # feature_tuples = [tuple(row.tolist()) for row in feature_loc]
                node_weights = explanation.node_mask[explanation.node_mask != 0]

                # print(node_weights.shape)
                if run_id == 0:
                    df = pd.DataFrame({
                        'node': feature_loc[:, 0].numpy(),
                        'feature': feature_loc[:, 1].numpy(),
                        f'weight_{run_id}': node_weights.numpy()
                    })
                else:
                    df[f'weight_{run_id}'] = node_weights.numpy()
                # df = pd.concat([df, new_df], ignore_index=True) # would add a row, but we want columns

            # Neighboring node importance (edge masks) ######################################################################
                mask_idx = np.where(explanation.edge_mask > 0)[0]
                edge_weights = explanation.edge_mask[mask_idx]
                edge_idx = explanation.edge_index[:, mask_idx].squeeze(1)
                transposed = edge_idx.transpose(0, 1)

                #print(transposed.shape)
                tuples = [tuple(row.tolist()) for row in transposed]

                for i in range(len(tuples)):
                    weight = edge_weights[i].item()
                    key = tuples[i]
                    if key in weights_dict:
                        weights_dict[key].append(weight)
                    else:
                        weights_dict[key] = [weight]
                # print(weights_dict)
            mean_edge_weights = {edge: sum(weights)/len(weights) for edge, weights in weights_dict.items()}
            df['mean']= df[['weight_0', 'weight_1', 'weight_2', 'weight_3', 'weight_4']].mean(axis=1)
            print(mean_edge_weights)
            print(tuples)
            print(df)

            feature_names, grouped = get_feature_list()

            for mode in ['single']:
                if mode == "single":
                    for _, i in enumerate(neighbors):
                        if i == node_idx:
                            TITLE = f"Feature importance of node {i}\n"
                        else:
                            TITLE = f"Feature importance of node {i} \nin neighborhood of node {node_idx}"
                        sort_i = df[df['node']==i]
                        # sort_i['weight'] = sort_i[run_list].mean(axis=1)
                        sort_i = sort_i.sort_values(by='mean', ascending=False)
                        sort_i = sort_i.reset_index()
                        sort_i['feature_name'] = sort_i['feature'].apply(lambda i: feature_names[i])
                        sort_i
                        # %%
                        temperature_features = ['t2m_mean', 't2m_std', 'mn2t6_mean', 'mn2t6_std', 'mx2t6_mean', 'mx2t6_std']
                        alt_orog = ['station_altitude', 'model_orography']
                        # colors = ['orange' if feature in temperature_features else 'skyblue' for feature in sort_i['feature_name']]
                        colors = [
                            'orange' if feature in temperature_features else
                            'gray' if feature in alt_orog else
                            'blue'
                            for feature in sort_i['feature_name']
                        ]
                        plt.figure(figsize=(8, 6))
                        plt.bar(sort_i['feature_name'][:20], sort_i['mean'][:20], color=colors)
                        # plt.bar(sort_i['feature_name'], sort_i['weight_mean'], color=colors)
                        # plt.xlabel('Feature', fontsize=24)
                        plt.ylabel('Importance weight', fontsize=24)
                        plt.ylim([0, 1])
                        plt.title(TITLE, fontsize=24)
                        plt.xticks(rotation=45, ha='right', fontsize=14)
                        plt.yticks(fontsize=14)
                        # plt.grid(axis='y')
                        plt.tight_layout()
                        plt.savefig(os.path.join(FIGUREPATH, f"gnnx_features_{leadtime}_{node_idx}_{graph_name}_{d}_{mode}-{i}.pdf"),
                                    format='pdf', dpi=300, bbox_inches='tight')
                        # plt.show()

                        # retrieve position from dataframes['stations']
                        pos_dict = {row['station_id']: (row['lon'], row['lat']) for _, row in dataframes['stations'].iterrows()}
                        alt_dict = {row['station_id']: row['altitude'] for _, row in dataframes['stations'].iterrows()}
                        node_list = df['node'].unique().tolist()
                        this_alt_dict = {node: alt_dict[node] for node in node_list if node in alt_dict}
                        lons = [pos_dict[node][0] for node in node_list if node in pos_dict]
                        lats = [pos_dict[node][1] for node in node_list if node in pos_dict]

                        subG = nx.Graph()

                        for (u, v), weight in zip(tuples, mean_edge_weights):
                            subG.add_edge(u, v, weight=weight)

                        edges = list(mean_edge_weights.keys())
                        e_weights = list(mean_edge_weights.values())

                        fig = plt.figure(figsize=(12, 12))
                        ax = plot_map()

                        cmap_alt = cm.terrain
                        cmap_edge = LinearSegmentedColormap.from_list("white_to_blue", ["white", "darkblue"])

                        norm_edge = plt.Normalize(0, 1)
                        norm_alt = plt.Normalize(vmin=min(dataframes['stations']['altitude']), vmax=max(dataframes['stations']['altitude']))

                        edge_colors = [cmap_edge(norm_edge(w)) for w in e_weights]
                        node_colors = [cmap_alt(norm_alt(this_alt_dict[node])) for node in subG.nodes()]

                        proj = ccrs.PlateCarree()
                        margin = 0.3
                        lon_min, lon_max = min(lons) - margin, max(lons) + margin
                        lat_min, lat_max = min(lats) - margin, max(lats) + margin

                        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
                        # ax.set_extent([2, 8, 50, 54], crs=proj)

                        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_alt, norm=norm_alt)
                        sm_edges = plt.cm.ScalarMappable(cmap=cmap_edge, norm=norm_edge)
                        colbar2 = plt.colorbar(sm_edges, ax=ax, aspect=30, pad=0.04)
                        colbar1 = plt.colorbar(sm_nodes, ax=ax, aspect=30, pad=0.04)

                        colbar2.ax.set_ylabel("Edge weights", rotation=270, labelpad=30, fontsize=24)
                        colbar2.ax.tick_params(labelsize=14)
                        colbar1.ax.set_ylabel("Altitude", rotation=270, labelpad=30, fontsize=24)
                        colbar1.ax.tick_params(labelsize=14)

                        nx.draw_networkx(subG, pos_dict, node_size=400, node_color=node_colors, ax=ax, edge_color=edge_colors,
                                         with_labels=False)
                        nx.draw_networkx_labels(subG, pos_dict, font_size=14)
                        plt.savefig(os.path.join(FIGUREPATH, f"gnnx_map_{leadtime}_{node_idx}_{graph_name}_{d}.pdf"), format='pdf', dpi=300,
                                    bbox_inches='tight')
                        plt.close('all')
                elif mode == 'all':
                    i = node_idx
                    sort_i = df.groupby('feature').mean()
                    # print(sort_i)
                    # sort_i['weight'] = sort_i[run_list].mean(axis=1)
                    sort_i = sort_i.sort_values(by='mean', ascending=False)
                    TITLE = f"Feature importance of all nodes \nin neighborhood of node {node_idx}"

                    sort_i = sort_i.reset_index()
                    sort_i['feature_name'] = sort_i['feature'].apply(lambda i: feature_names[i])
                    sort_i
                    # %%
                    temperature_features = ['t2m_mean', 't2m_std', 'mn2t6_mean', 'mn2t6_std', 'mx2t6_mean', 'mx2t6_std']
                    alt_orog = ['station_altitude', 'model_orography']
                    # colors = ['orange' if feature in temperature_features else 'skyblue' for feature in sort_i['feature_name']]
                    colors = [
                        'orange' if feature in temperature_features else
                        'gray' if feature in alt_orog else
                        'blue'
                        for feature in sort_i['feature_name']
                    ]
                    plt.figure(figsize=(8, 6))
                    plt.bar(sort_i['feature_name'][:20], sort_i['mean'][:20], color=colors)
                    # plt.bar(sort_i['feature_name'], sort_i['weight_mean'], color=colors)
                    # plt.xlabel('Feature', fontsize=24)
                    plt.ylabel('Importance weight', fontsize=24)
                    plt.ylim([0, 1])
                    plt.title(TITLE, fontsize=24)
                    plt.xticks(rotation=45, ha='right', fontsize=14)
                    plt.yticks(fontsize=14)
                    # plt.grid(axis='y')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(FIGUREPATH, f"gnnx_features_{leadtime}_{node_idx}_{graph_name}_{d}_{mode}-{i}.pdf"),
                        format='pdf', dpi=300, bbox_inches='tight')
                    # plt.show()

                    # retrieve position from dataframes['stations']
                    pos_dict = {row['station_id']: (row['lon'], row['lat']) for _, row in dataframes['stations'].iterrows()}
                    alt_dict = {row['station_id']: row['altitude'] for _, row in dataframes['stations'].iterrows()}
                    node_list = df['node'].unique().tolist()
                    this_alt_dict = {node: alt_dict[node] for node in node_list if node in alt_dict}
                    lons = [pos_dict[node][0] for node in node_list if node in pos_dict]
                    lats = [pos_dict[node][1] for node in node_list if node in pos_dict]
                    this_alt_dict
                    # %%

                    subG = nx.Graph()

                    for (u, v), weight in zip(tuples, mean_edge_weights):
                        subG.add_edge(u, v, weight=weight)

                    edges = list(mean_edge_weights.keys())
                    e_weights = list(mean_edge_weights.values())

                    fig = plt.figure(figsize=(12, 12))
                    ax = plot_map()

                    cmap_alt = cm.terrain
                    cmap_edge = LinearSegmentedColormap.from_list("white_to_blue", ["white", "darkblue"])

                    norm_edge = plt.Normalize(0, 1)
                    norm_alt = plt.Normalize(vmin=min(dataframes['stations']['altitude']),
                                             vmax=max(dataframes['stations']['altitude']))

                    edge_colors = [cmap_edge(norm_edge(w)) for w in e_weights]
                    node_colors = [cmap_alt(norm_alt(this_alt_dict[node])) for node in subG.nodes()]

                    proj = ccrs.PlateCarree()
                    margin = 0.3
                    lon_min, lon_max = min(lons) - margin, max(lons) + margin
                    lat_min, lat_max = min(lats) - margin, max(lats) + margin

                    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
                    # ax.set_extent([2, 8, 50, 54], crs=proj)

                    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_alt, norm=norm_alt)
                    sm_edges = plt.cm.ScalarMappable(cmap=cmap_edge, norm=norm_edge)
                    colbar2 = plt.colorbar(sm_edges, ax=ax, aspect=30, pad=0.04)
                    colbar1 = plt.colorbar(sm_nodes, ax=ax, aspect=30, pad=0.04)

                    colbar2.ax.set_ylabel("Edge weights", rotation=270, labelpad=30, fontsize=24)
                    colbar2.ax.tick_params(labelsize=14)
                    colbar1.ax.set_ylabel("Altitude", rotation=270, labelpad=30, fontsize=24)
                    colbar1.ax.tick_params(labelsize=14)

                    nx.draw_networkx(subG, pos_dict, node_size=400, node_color=node_colors, ax=ax, edge_color=edge_colors,
                                     with_labels=False)
                    nx.draw_networkx_labels(subG, pos_dict, font_size=14, font_color='white')
                    plt.savefig(os.path.join(FIGUREPATH, f"gnnx_map_{leadtime}_{node_idx}_{graph_name}_{d}.pdf"), format='pdf',
                                dpi=300, bbox_inches='tight')
                    plt.close()

                else:
                    print("Not a valid input.")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process a graph by name.")
    parser.add_argument(
        "--graph_name",
        type=str,
        default="g1",
        help="Name of the graph to process (default: g1)"
    )

    args = parser.parse_args()
    main(args.graph_name)

