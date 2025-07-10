import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cmasher as cmr
from torch_geometric.data import Data

def plot_map() -> plt.Axes:
    """
    Plot a map using PlateCarree projection.

    Returns:
        The Axes object containing the map.
    """
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([1, 12, 45, 54], crs=proj)
    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))
    return ax

