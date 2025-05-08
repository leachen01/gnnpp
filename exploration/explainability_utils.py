import torch

def get_feature_list(test_df):
    feature_names = [f for f in test_df.columns.tolist() if f not in ['time', 'number']]
    grouped = []
    i = 0
    while i < len(feature_names):
        if feature_names[i] == 'cos_doy':
            if i + 1 < len(feature_names) and 'sin_doy' in feature_names[i+1]:
                grouped.append([feature_names[i], feature_names[i+1]])
                i += 2
            else:
                grouped.append([feature_names[i]])
                i += 1
        elif i + 1 < len(feature_names) and feature_names[i+1] == feature_names[i].replace('mean', 'std'):
            grouped.append([feature_names[i], feature_names[i+1]])
            i += 2
        else:
            grouped.append([feature_names[i]])
            i += 1
    return feature_names, grouped

def shuffle_drn_features(xs, feature_permute_idx):
    xs_permuted = xs[..., feature_permute_idx]  # (86742,)
    print(xs_permuted.shape) # [86742, 1]

    T = xs_permuted.shape[0] # T,
    perm_T = torch.randperm(T)  # permute the features in column across time
    xs_permuted = xs_permuted[perm_T, ...]
    # print(xs_permuted.shape)

    # Replace features with permuted features
    xs[..., feature_permute_idx] = xs_permuted
    return xs