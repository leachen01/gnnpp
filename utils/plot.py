import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
    ax.coastlines()
    ax.set_extent([1, 12, 45, 54], crs=proj)
    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))
    return ax


def visualize_graph(d: Data) -> None:
    """
    Visualize the Generated Data as a graph.

    :param d: torch_geometric data object
    :type d: Data
    """
    plt.style.use("default")
    if d.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # to cpu for further calculations and plotting
    edge_index, dist = d.edge_index.cpu().numpy(), d.edge_attr.cpu().numpy()  # Was distances before

    # NOTE: edge_index_att holds the Edges of the new graph,
    # however they are labeled consecutively instead of the ordering from stations DataFrame
    station_ids = np.array(d.x[:, 0])
    edge_index = station_ids[edge_index]  # now the same indexes as in the stations Dataframe are used

    # Add nodes (stations) to the graph
    for i in range(d.num_nodes):
        G.add_node(int(d.x[i, 0]), lon=float(d.pos[i, 0]), lat=float(d.pos[i, 1]))  # Add station with ID, LAT and LON

    pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}  # Create a positions dict

    # Add edges with edge_length as an attribute
    if d.is_directed():
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):  # Add all edges
            G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute
    else:
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):
            if not (G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0])):  # Edge only needs to be added once
                G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute

    # Colors
    degrees = G.degree if d.is_undirected() else G.in_degree

    node_colors = [deg for _, deg in degrees]
    cmap_nodes = plt.get_cmap("jet", max(node_colors) - min(node_colors) + 1)
    norm = plt.Normalize(min(node_colors), max(node_colors))
    colors_n = [cmap_nodes(norm(value)) for value in node_colors]
    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm)

    # Edge Colors
    color_values = [attr["length"] for _, _, attr in G.edges(data=True)]
    cmap = mpl.colormaps.get_cmap("Blues_r")
    # Normalize the values to range between 0 and 1
    norm = plt.Normalize(min(color_values), max(color_values))
    # Generate a list of colors based on the normalized values
    colors = [cmap(norm(value)) for value in color_values]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot map
    ax = plot_map()

    # Colorbar Node degrees
    if not all(node_colors[0] == col for col in node_colors[1:]):  # only add colorbar if there are different degrees
        colorbar = plt.colorbar(sm_nodes, ax=ax)
        ticks_pos = (
            np.linspace(
                min(node_colors) + 1,
                max(node_colors),
                max(node_colors) - min(node_colors) + 1,
            )
            - 0.5
        )
        colorbar.set_ticks(ticks_pos)
        ticks = np.arange(min(node_colors), max(node_colors) + 1)
        colorbar.set_ticklabels(ticks)
        colorbar.ax.set_ylabel(f'Node{"_in" if d.is_directed() else ""} Degree', rotation=270, labelpad=20)

    # Colormap for Edges
    colorbar_e = plt.colorbar(sm, ax=ax)
    colorbar_e.ax.set_ylabel("Normalized Distance", rotation=270, labelpad=20)

    # Plot Graph
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=20,
        node_color=colors_n,
        ax=ax,
        with_labels=False,
        edge_color=colors,
        edgecolors="black",
    )

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))

    ax.set_title("Active weather stations in Germany")
    plt.savefig("stations.eps", format="eps")
    plt.show()
