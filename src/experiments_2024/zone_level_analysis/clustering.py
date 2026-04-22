import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from experiments_2024.zone_level_analysis.base import input_to_dict


def run_1D_clustering(ser, slices, percentiles=True, mapping=None):
    """
    Manual clustering of a single var into buckets along its axis

    Parameters
    ----------
    ser : pd.Series
        data to slice
    slices : list
        either percentiles or values
    percentiles : bool
        If True, slices interpreted as percentiles
        If False, interpreted as values
    mapping : dict
        ability to map groups
        e.g. {0 : 0, 1 : 1, 2 : 1, 3 : 1}

    Returns
    -------
    pd.Series with group numbers
    """
    # make compatible with df
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]

    zones = list(ser.index)
    slice_groups = pd.Series(np.nan, index=zones)

    # check slices
    slices = copy.deepcopy(slices)
    if percentiles:
        if 1 not in slices:
            slices.append(1)
        if 0 not in slices:
            slices.append(0)
    else:
        if ser.max() not in slices:
            slices.append(ser.max())
        if ser.min() not in slices:
            slices.append(ser.min())
    slices.sort(reverse=True)

    # slices to boundaries
    if percentiles:
        boundaries = []
        for slice in slices:
            boundaries.append(ser.quantile(slice))
    else:
        boundaries = slices

    # run slicing
    for i in range(len(boundaries) - 1):
        these_zones = list((ser[ser <= boundaries[i]][ser >= boundaries[i + 1]]).index)
        if mapping is not None and i in mapping:
            group = mapping[i]
        else:
            group = i
        slice_groups[these_zones] = group
    return slice_groups


def run_2D_clustering(df, var1_slice, var2_slice, mapping=None):
    """
    Manual clustering of two vars, each limited to one slice (divide into 4 quadrants)

    Parameters
    ----------
    df : pd.DataFrame
        df with index as zones and columns as tests
    var1_slice : float
        where to place var1 slice
    var2_slice : float
        where to place var2 slice
    mapping : dict
        ability to map quandrants to groups
        e.g. {0 : 0, 1 : 1, 2 : 1, 3 : 1}

    Returns
    -------
    pd.Series with group numbers
    if var1 is y and var2 is x, then:
    0 is upper right quadrant
    1 is upper left
    2 is lower left
    3 is lower right
    4 is NAN
    """
    zones = list(df.index)
    cols = list(df.columns)
    clusters = pd.Series(np.nan, index=zones)
    if len(cols) > 2:
        print("Only support manual clustering of two columns")
        return clusters
    var1 = cols[0]
    var2 = cols[1]
    # unify map
    if mapping is None:
        mapping = {}
    for i in range(5):
        if i not in mapping:
            mapping[i] = i
    for zone in zones:
        if df.loc[zone, var1] >= var1_slice and df.loc[zone, var2] >= var2_slice:
            clusters[zone] = mapping[0]
        if df.loc[zone, var1] >= var1_slice and df.loc[zone, var2] < var2_slice:
            clusters[zone] = mapping[1]
        if df.loc[zone, var1] < var1_slice and df.loc[zone, var2] < var2_slice:
            clusters[zone] = mapping[2]
        if df.loc[zone, var1] < var1_slice and df.loc[zone, var2] >= var2_slice:
            clusters[zone] = mapping[3]
    clusters.fillna(mapping[4], inplace=True)
    return clusters


def run_kmeans_clustering(df, n_clusters, mapping=None):
    """
    Runs a k-means clustering over df with zones as index and columns as tests

    Parameters
    ----------
    df : pd.DataFrame
        df to run clustering on
    n_clusters : int
        number of clusters
    mapping : dict
        can map cluster groups after clustering
        {1 : 0} maps cluster 1 to 0

    Returns
    -------
    tuple (pd.Series with clustering, np.array containing cluster center information)
    """
    df.dropna(inplace=True)
    # run cluster model
    clusters_model = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    # rearrange labels so they are in ascending order of norm(centroid)
    df["Old Cluster Group"] = clusters_model.labels_
    df["New Cluster Group"] = np.nan
    cluster_map = pd.DataFrame(index=range(n_clusters), columns=["Sort Value"])
    cluster_map["Sort Value"] = np.sum(clusters_model.cluster_centers_, axis=1)
    cluster_map = cluster_map.sort_values("Sort Value")
    cluster_map["New Cluster Group"] = range(len(list(cluster_map.index)))
    zones = list(df.index)
    for i in range(len(zones)):
        zone = zones[i]
        old_cluster = df.loc[zone, "Old Cluster Group"]
        df.loc[zone, "New Cluster Group"] = int(
            cluster_map.loc[old_cluster, "New Cluster Group"]
        )
    result = df["New Cluster Group"]
    # fill nas (always last group)
    max_cluster = result.max()
    result.fillna(max_cluster + 1, inplace=True)
    max_cluster = result.max()
    # optional map
    if mapping is not None:
        result = result.replace(mapping)
    # save cluster centers
    cluster_centers = np.zeros(clusters_model.cluster_centers_.shape)
    for i in range(cluster_centers.shape[0]):
        row = cluster_map.index[i]
        cluster_centers[i, :] = clusters_model.cluster_centers_[row, :]
    return result, cluster_centers


def run_1D_clustering_on_dict(
    this_dict, slices, col=None, percentiles=False, mapping=None
):
    """
    Runs 1D clustering on several buildings saved in a dict

    Parameters
    ----------
    this_dict : dict
        dict with building as key, df as values with summary results
    slices : list or dict
        if list, same list of slices applied to all buildings
        if dict, can apply specific slices (list) to each building
    col : str
        column in this_dict to cluster
        default = None, which means use first column
    percentiles : bool or dict
        whether slices are interpreted as values or percentiles
        default = False (interpret as values)
        if bool, same logic applies to all buildings
        if dict, can be building specific
    mapping : dict
        ability to map groups
        e.g. {0 : 0, 1 : 1, 2 : 1, 3 : 1}

    Returns
    -------
    dict with clusters
    """
    # prep inputs
    projects = list(this_dict.keys())
    slices_dict = input_to_dict(slices, projects)
    percentiles_dict = input_to_dict(percentiles, projects)
    if col is None:
        col = list(this_dict[projects[0]].columns)[0]
    cluster_dict = {}
    for project in projects:
        clusters = run_1D_clustering(
            this_dict[project][col],
            slices_dict[project],
            percentiles=percentiles_dict[project],
            mapping=mapping,
        ).to_frame()
        cluster_dict[project] = clusters
    return cluster_dict


def run_2D_clustering_on_dict(
    this_dict1,
    this_dict2,
    slice1,
    slice2,
    col1=None,
    col2=None,
    mapping=None,
):
    """
    Runs 2D clustering on several buildings saved in a results dicts

    Parameters
    ----------
    this_dict1 : dict
        dict with building as key, df as values with summary results
    this_dict2 : dict
        dict with building as key, df as values with summary results
    slice1 : float or dict
        where to place slice in variable 1, interpreted as values
        if float, same slice applied to all buildings
        if dict, can be building specific
    slice2 : float or dict
        where to place slice in variable 2, interpreted as values
        if float, same slice applied to all buildings
        if dict, can be building specific
    col1 : str
        column to use from results1_dict
        default = None, which means use first column
    col2 : str
        column to use from results2_dict
        default = None, which means use first column
    mapping : dict
        can map cluster groups after clustering
        {1 : 0} maps cluster 1 to 0
        input type can be a dict, in which case same map applied to all
        or can be a dict of dicts, in which case uses can apply building specific maps

    Returns
    -------
    dict with clusters
    """
    projects = list(this_dict1.keys())
    slice1_dict = input_to_dict(slice1, projects)
    slice2_dict = input_to_dict(slice2, projects)
    if (mapping is None) or (not (isinstance(mapping[list(mapping.keys())[0]], dict))):
        mapping_dict = {}
        for project in projects:
            mapping_dict[project] = mapping
    else:
        mapping_dict = mapping

    if col1 is None:
        col1 = list(this_dict1[projects[0]].columns)[0]
    if col2 is None:
        col2 = list(this_dict2[projects[0]].columns)[0]
    cluster_dict = {}
    for project in projects:
        df = pd.concat([this_dict1[project][col1], this_dict2[project][col2]], axis=1)
        clusters = run_2D_clustering(
            df,
            slice1_dict[project],
            slice2_dict[project],
            mapping=mapping_dict[project],
        ).to_frame()
        cluster_dict[project] = clusters
    return cluster_dict


def run_kmeans_clustering_on_dict(this_dict, n_clusters, cols=None, mapping=None):
    """
    Runs kmeans clustering on several buildings saved in a results dicts

    Parameters
    ----------
    this_dict : dict
        dict with building as key, df as values with summary results
    n_clusters : int or dict
        if int, same n_clusters used for all buildings
        if dict, can be building specific
    cols : list
        list of columns to run clustering on in this_dict
        default = None, which means use all columns
    mapping : dict
        can map cluster groups after clustering
        {1 : 0} maps cluster 1 to 0
        input type can be a dict, in which case same map applied to all
        or can be a dict of dicts, in which case uses can apply building specific maps

    Returns
    -------
    dict with clusters
    """
    projects = list(this_dict.keys())
    n_clusters_dict = input_to_dict(n_clusters, projects)
    if (mapping is None) or not ((isinstance(mapping[list(mapping.keys())[0]], dict))):
        mapping_dict = {}
        for project in projects:
            mapping_dict[project] = mapping
    else:
        mapping_dict = mapping

    if cols is None:
        cols = list(this_dict[projects[0]].columns)

    cluster_dict = {}
    for project in projects:
        df = this_dict[project][cols]
        clusters = run_kmeans_clustering(
            df,
            n_clusters_dict[project],
            mapping_dict[project],
        )[0].to_frame()
        cluster_dict[project] = clusters
    return cluster_dict
