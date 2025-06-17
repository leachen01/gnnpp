import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


def get_feature_list(test_df=None):
    if test_df == None:
        feature_names = ['station_id', 'model_orography', 'station_altitude', 'station_latitude',
                         'station_longitude', 'cape_mean', 'cape_std', 'sd_mean', 'sd_std', 'stl1_mean', 'stl1_std', 'swvl1_mean',
                         'swvl1_std', 't2m_mean', 't2m_std', 'tcc_mean', 'tcc_std', 'tcw_mean', 'tcw_std',
                         'tcwv_mean', 'tcwv_std', 'u10_mean', 'u10_std', 'u100_mean', 'u100_std', 'v10_mean', 'v10_std',
                         'v100_mean', 'v100_std', 'vis_mean', 'vis_std', 'cp6_mean', 'cp6_std', 'mn2t6_mean', 'mn2t6_std', 'mx2t6_mean', 'mx2t6_std', 'p10fg6_mean',
                         'p10fg6_std', 'slhf6_mean', 'slhf6_std', 'sshf6_mean', 'sshf6_std', 'ssr6_mean', 'ssr6_std', 'ssrd6_mean', 'ssrd6_std',
                         'str6_mean', 'str6_std', 'strd6_mean', 'strd6_std', 'tp6_mean', 'tp6_std', 'z_mean',
                         'z_std', 'q_mean', 'q_std', 'u_mean', 'u_std', 'v_mean', 'v_std', 't_mean', 't_std', 'cos_doy',
                         'sin_doy']
    else:
        feature_names = [f for f in test_df.columns.tolist() if
                         f not in ['time', 'number']]  # dropped time and number: len - 65

    grouped = []
    i = 0
    while i < len(feature_names):
        if feature_names[i] == 'cos_doy':
            if i + 1 < len(feature_names) and 'sin_doy' in feature_names[i + 1]:
                grouped.append([feature_names[i], feature_names[i + 1]]) #
                # grouped.append('doy')
                i += 2
            else:
                grouped.append(feature_names[i])
                i += 1
        elif i + 1 < len(feature_names) and feature_names[i + 1] == feature_names[i].replace('mean', 'std'):
            grouped.append([feature_names[i], feature_names[i + 1]]) #
            # grouped.append(feature_names[i].split("_")[0])
            i += 2
        else:
            grouped.append(feature_names[i])
            i += 1
    return feature_names, grouped


def shuffle_drn_features(xs, feature_permute_idx):
    xs_permuted = xs[..., feature_permute_idx]  # (86742,)
    print(xs_permuted.shape)  # [86742, 1]

    T = xs_permuted.shape[0]  # T,
    perm_T = torch.randperm(T)  # permute the features in column across time
    xs_permuted = xs_permuted[perm_T, ...]
    # print(xs_permuted.shape)

    # Replace features with permuted features
    xs[..., feature_permute_idx] = xs_permuted
    return xs


class MultigraphWrapper(nn.Module):
    def __init__(self, model):
        super(MultigraphWrapper, self).__init__()
        self.model = model
        self.training = model.training
        self.device = next(model.parameters()).device
        self.edge_dim = getattr(model, 'edge_dim', 1)
        # print(self.device)

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        # print(f"x.shape: {x.shape}, device: {x.device}")
        # print(f"edge_index.shape: {edge_index.shape}, device: {edge_index.device}")
        # print(f"edge_attr.shape: {edge_attr.shape}, device: {edge_attr.device}")
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        data.n_idx = torch.arange(x.size(0), device=self.device)
        return self.model(data)


def create_explainer(model, epochs, lr, edge_size: float=0.005, node_feat_size: float= 1.0):
    # print(model)
    wrapped_model = MultigraphWrapper(model)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(
            epochs=epochs,
            lr=lr,
            coeffs = {
                'edge_size': edge_size,
                'node_feat_size': node_feat_size,
            }),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='regression',
            task_level='node',
            return_type='raw'  # crps? globale wichtigkeit
        )
    )
    return explainer
