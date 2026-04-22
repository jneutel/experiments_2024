import copy
import pandas as pd
import functools


MW_PER_TON = 3.5168528421 / 1000
WH_PER_BTU = 0.293071
M2_PER_SF = 0.092903

ALL_BUILDINGS = [
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
]

EXPERIMENTS_BUILDINGS = {
    "2024": [
        "OFF-2",
        "OFF-3",
        "OFF-4",
        "OFF-5",
        "OFF-6",
        "OFF-7",
    ],
}


def input_to_dict(input, projects):
    """
    Helper function to check whether input is of dict type, and if not converts it

    Parameters
    ----------
    input : many
        an input to another function, can be int, str etc
    projects : list
        list of buildings to convert input_dict to, with buildings as key

    Returns
    -------
    dict ready to be input to function
    """
    if isinstance(input, dict):
        return input
    return_dict = {}
    for project in projects:
        return_dict[project] = input
    return return_dict


def trim_to_common_elements(dfs, clean_cols=True, clean_idx=True):
    """
    Trim a list of dataframe to only include common elements and indexes

    Parameters
    ----------
    dfs : list of pd.DataFrames
    clean_cols : bool
    clean_idx :  bool

    Returns
    -------
    list of pandas.DataFrames
    """
    for i in range(len(dfs)):
        dfs[i] = copy.deepcopy(dfs[i])
    if len(dfs) == 1:
        return dfs
    if clean_cols:
        cols = functools.reduce(
            lambda left, right: left.intersection(right),
            [dfs[i].columns for i in range(len(dfs))],
        )
        dfs = [df.loc[:, cols] for df in dfs]
    if clean_idx:
        idx = functools.reduce(
            lambda left, right: left.intersection(right),
            [dfs[i].index for i in range(len(dfs))],
        )
        dfs = [df.loc[idx, :] for df in dfs]
    return dfs


def make_common_index(dicts, print_message=False):
    """
    Helper function to make sure dicts have common indexes

    Parameters
    ----------
    dicts : list
        list of dicts, which are dict of pd.DataFrame()'s
    print_message : bool
        whether or not to print out which zones are lost by the operation

    Returns
    -------
    list
    """
    for project in dicts[0]:
        dfs = []
        for this_dict in dicts:
            dfs.append(this_dict[project])
        newdfs = trim_to_common_elements(dfs, clean_cols=False, clean_idx=True)
        if print_message:
            for i in range(len(dfs)):
                lost_boxes = list(set(list(dfs[i].index)) - set(list(newdfs[i].index)))
                if len(lost_boxes) > 0:
                    print(
                        f"By making common we have lost these boxes in {project}: {lost_boxes}"
                    )
        i = 0
        for this_dict in dicts:
            this_dict[project] = newdfs[i]
            i += 1
    return dicts


def run_passive_test(df, this_test, axis=0):
    if this_test == "Sum":
        return df.sum(axis=axis)
    if this_test == "Mean":
        return df.mean(axis=axis)
    if this_test == "Median":
        return df.median(axis=axis)
    if this_test == "Min":
        return df.min(axis=axis)
    if this_test == "Max":
        return df.max(axis=axis)
    if this_test == "Std":
        return df.std(axis=axis)


def run_passive_test_on_dfs(
    dfs,
    this_test="Mean",
    col_name=None,
    results=None,
):
    """
    Helper function to run passive test and return dict object ready for plotting

    Parameters
    ----------
    dfs : dict
        building as key and df as value
    this_test : str
        statistic to summarize over time
        Options include "Sum", "Mean", "Median", "Min", "Max", "Std"
        default Mean
    col_name : str
        default None, name of test is used
    results : dict
        function accepts its own output so that results dict's can be built up

    Returns
    -------
    dict of pd.DataFrame()'s
    """
    projects = list(dfs.keys())
    # init
    if results is None:
        results = {}
        for project in projects:
            results[project] = pd.DataFrame()
    # col name
    if col_name is None:
        col_name = this_test
    for project in projects:
        df = dfs[project]
        # run test
        this_result = run_passive_test(df, this_test, axis=0).to_frame()
        this_result.sort_index(inplace=True)
        this_result.columns = [col_name]
        results[project] = pd.concat([results[project], this_result], axis=1)
    return results


# Operations on data inputs


def run_ahu_to_vav(df, map, airflow=None):
    """
    Helper function to convert df with AHU columns to VAV columns (index time)

    Parameters
    ----------
    df : pd.DataFrame() or dict(pd.DataFrame())
        input can be df or dict of dfs
        df with index as time and columns as VAVs
    map : pd.Series()
        input can be series or dict of series
        ser with index as VAVs and value as AHU
    airflow : pd.DataFrame() or dict(pd.DataFrame())
        default None (optional input)
        provided, we divy up AHU value based on zone airflow
        not provided, just copy AHU value
        input can be df or dict of dfs
        df with index as time and columns as VAVs

    Returns
    -------
    df or dict(df) with index time, VAV columns
    """
    dfs = input_to_dict(df, ["dummy"])
    maps = input_to_dict(map, ["dummy"])
    if airflow is not None:
        airflows = input_to_dict(airflow, ["dummy"])
    keys = list(dfs.keys())

    df_vavs = {}
    for key in keys:
        this_df = dfs[key]
        this_map = maps[key]

        df_vav = pd.DataFrame(index=this_df.index)
        for ahu in this_df.columns:
            if ahu in this_map.values:
                zones = list(this_map[this_map == ahu].dropna().index)
                if airflow is not None:
                    zones = list(set(zones).intersection(set(airflows[key].columns)))
                    this_airflow = airflows[key][zones]
                    this_airflow_norm = this_airflow.div(
                        this_airflow.sum(axis=1), axis=0
                    )  # norm
                    for zone in zones:
                        df_vav[zone] = this_df[ahu] * this_airflow_norm[zone]
                else:
                    for zone in zones:
                        df_vav[zone] = this_df[ahu]
        df_vavs[key] = df_vav

    if len(keys) == 1:
        return df_vavs[keys[0]]
    return df_vavs


def run_vav_to_ahu(df, map, vav_to_ahu="Mean"):
    """
    Helper function to convert df with VAV columns to AHU columns (index time)

    Parameters
    ----------
    df : pd.DataFrame()
        input can be df or dict of dfs
        df with index as time and columns as VAVs
    map : pd.Series()
        input can be series or dict of series
        ser with index as VAVs and value as AHU
    vav_to_ahu : str
        how to summarize by AHU, "Sum", "Mean", "Median", "Min", "Max"
        default Mean

    Returns
    -------
    df or dict(df) with index time, AHU columns
    """
    dfs = input_to_dict(df, ["dummy"])
    maps = input_to_dict(map, ["dummy"])
    keys = list(dfs.keys())

    df_ahus = {}
    for key in keys:
        map = maps[key]
        df = dfs[key]
        ahus = list(map.unique())
        df_ahu = pd.DataFrame(index=df.index, columns=ahus)
        for ahu in ahus:
            df_temp = df.loc[:, df.columns.isin(list((map[map == ahu]).index))]
            df_ahu[ahu] = run_passive_test(df_temp, vav_to_ahu, axis=1)
        df_ahus[key] = df_ahu

    if len(keys) == 1:
        return df_ahus[keys[0]]
    return df_ahus


def run_vav_to_room(df, map, vav_to_room="Mean"):
    """
    Helper function to convert df with VAV columns to room columns (index time)

    Parameters
    ----------
    df : pd.DataFrame() or dict(pd.DataFrame())
        input can be df or dict of dfs
        df with index as time and columns as VAVs
    map : pd.Series()
        input can be series or dict of series
        ser with index as VAVs and value as room
    vav_to_room : str
        how to summarize by room, "Sum", "Mean"
        default Mean

    Returns
    -------
    df or dict(df) with index time, room columns
    """
    dfs = input_to_dict(df, ["dummy"])
    maps = input_to_dict(map, ["dummy"])
    keys = list(dfs.keys())

    df_rooms = {}
    for key in keys:
        df = copy.deepcopy(dfs[key]).T
        map = maps[key]
        df["Room"] = df.index.map(map)
        if vav_to_room == "Mean":
            df = df.groupby("Room").mean().T
        elif vav_to_room == "Sum":
            df = df.groupby("Room").sum().T
        df_rooms[key] = df

    if len(keys) == 1:
        return df_rooms[keys[0]]
    return df_rooms


def run_vavs_to_buiding_ahus(dfs, maps, vav_to_ahu="Mean"):
    """
    Helper function to convert dfs with VAV columns and time index to one df with building_ahu columns

    Parameters
    ----------
    dfs : dict of pd.DataFrame()
        dict with key as building and value as df
    maps : dict of pd.Series()
        dict with building as key, value is pd.Series() with index as VAVs and value as AHU
    vav_to_ahu : str
        how to summarize by AHU, "Sum" or "Mean"
        default Mean

    Returns
    -------
    pd.DataFrame() with index time, building columns
    """
    projects = list(dfs.keys())
    building_ahu_df = pd.DataFrame()
    for project in projects:
        df = dfs[project]
        df = run_vav_to_ahu(df, maps[project], vav_to_ahu)
        new_cols = [f"{project} {col}" for col in list(df.columns)]
        df.columns = new_cols
        building_ahu_df = pd.concat([building_ahu_df, df], axis=1)
    building_ahu_df = building_ahu_df.reindex(sorted(building_ahu_df.columns), axis=1)
    building_ahu_df.index = pd.DatetimeIndex(building_ahu_df.index)
    return building_ahu_df


def run_vavs_to_buidings(dfs, vav_to_building="Mean"):
    """
    Helper function to convert dfs with VAV columns and time index to one df with building columns

    Parameters
    ----------
    dfs : dict of pd.DataFrame()
        dict with key as building and value as df
    vav_to_building : str
        how to summarize by building, "Sum", "Mean", "Median", "Min", "Max"
        default Mean

    Returns
    -------
    pd.DataFrame() with index time, building columns
    """
    projects = list(dfs.keys())
    building_df = pd.DataFrame(index=dfs[projects[0]].index)
    for project in projects:
        df = dfs[project]
        ser = run_passive_test(df, vav_to_building, axis=1)
        building_df[project] = ser
    return building_df


def calculate_airflow_weighted_average(this_dict, airflow_dict, project="dummy"):
    """
    Helper function to calculate airflow weighted average

    Parameters
    ----------
    this_dict : dict or pd.DataFrame
        dict with building as key and pd.DataFrame as value
        compatible with single df with "dummy" key
    airflow_dict : dict
        dict with building as key and pd.DataFrame as value
        should have some keys, columns, index as this_dict
        compatible with single df with "dummy" key
    project : str
        optional, to help make compatible with single df

    Returns
    -------
    weighted_average_dict
    """
    this_dict = input_to_dict(this_dict, [project])
    airflow_dict = input_to_dict(airflow_dict, [project])

    projects = list(this_dict.keys())
    weighted_average_dict = {}

    for project in projects:
        this_dict[project] = this_dict[project].dropna(how="all", axis=1)
        airflow_dict[project] = airflow_dict[project].dropna(how="all", axis=1)
        cols = list(
            set(this_dict[project].columns).intersection(
                set(airflow_dict[project].columns)
            )
        )
        idx = list(
            set(this_dict[project].index).intersection(set(airflow_dict[project].index))
        )
        idx.sort()
        weighted_average_dict[project] = (
            (
                (
                    this_dict[project].loc[idx, cols]
                    * airflow_dict[project].loc[idx, cols]
                ).sum(axis=1)
            )
            / (airflow_dict[project].loc[idx, cols].sum(axis=1))
        ).to_frame()
        weighted_average_dict[project].columns = [project]
    return weighted_average_dict


# Operations on results dicts


def flip_dict(this_dict):
    """
    Helper function to convert flip dict keys and df columns

    Parameters
    ----------
    this_dict : dict
        dict with key as str and df as value

    Returns
    -------
    dict with pd.DataFrame value

    Notes
    -----
    Asssumes all dfs have same columns
    """

    keys = list(this_dict.keys())
    cols = list(this_dict[keys[0]].columns)
    idxs = {}

    for key in keys:
        for col in cols:
            if col not in idxs:
                idxs[col] = set()
            this_df = this_dict[key]
            idxs[col] = idxs[col].union(set(list(this_df.index)))

    for col in cols:
        idxs[col] = sorted(list(idxs[col]))

    new_dict = {}
    for key in keys:
        for col in cols:
            if col not in new_dict:
                new_dict[col] = pd.DataFrame(index=idxs[col], columns=keys)
            new_dict[col][key] = this_dict[key][col]
    return new_dict


def combine_dicts(dicts):
    """
    Helper function to combine several dicts into one dict with many columns

    Parameters
    ----------
    dicts : list
        list of results dicts

    Returns
    -------
    Combined dict
    """
    dicts = make_common_index(dicts, print_message=False)
    new_dict = {}
    for this_dict in dicts:
        for this_project in this_dict:
            this_df = this_dict[this_project]
            if this_project not in new_dict:
                new_dict[this_project] = pd.DataFrame(index=this_df.index)
            new_dict[this_project][list(this_df.columns)] = this_df
    return new_dict


def collapse_dict_to_df(this_dict, col_prefix=True):
    """
    Helper function to convert dict with project key and df value, to one df

    Parameters
    ----------
    this_dict : dict
    col_prefix : bool

    Returns
    -------
    pd.DataFrame()
    """
    df = pd.DataFrame()
    for project in this_dict:
        for col in this_dict[project]:
            if col_prefix:
                col_name = f"{project} {col}"
            else:
                col_name = project
            new_col = this_dict[project][col].to_frame()
            new_col.columns = [col_name]
            df = pd.concat([df, new_col], axis=1)
    return df


def combine_to_total(this_dict, buildings=None):
    """
    Helper function to append results from all buildings

    Parameters
    ----------
    this_dict : dict
        results dict
    buildings : list

    Returns
    -------
    df with all the results appended
    """
    if buildings is None:
        buildings = list(this_dict.keys())
    tot = []
    for building in buildings:
        if building not in this_dict:
            continue
        this_result = copy.deepcopy(this_dict[building])
        if this_result is None or this_result.empty:
            continue
        this_result.index = building + " " + this_result.index
        tot.append(this_result)
    tot = pd.concat(tot)
    return tot


def split_dict_into_ahus(this_dict, maps):
    """
    Helper function to split dict with project keys into building_ahu keys

    Parameters
    ----------
    this_dict : dict
        building as keys
    maps : dict
        dict of pd.Series(), with VAV as index and AHU as values

    Returns
    -------
    split dict
    """
    ahu_dict = {}
    for project in this_dict:
        map = maps[project]
        if isinstance(map, pd.DataFrame):
            map = map.iloc[:, 0]  # to series
        df = this_dict[project]
        ahus = list(map.unique())
        ahus.sort()
        for ahu in ahus:
            zones = list(map[map == ahu].index)
            zones = list(set(zones).intersection(set(list(df.index))))
            ahu_dict[f"{project} {ahu}"] = df.loc[zones, :]
    return ahu_dict
