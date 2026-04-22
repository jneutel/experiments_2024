import inspect
import numpy as np
import pandas as pd
import psychrolib
from experiments_2024 import DATASETS_PATH
from experiments_2024.datasets import utils
from experiments_2024.zone_level_analysis.cleaning import clean_df

psychrolib.SetUnitSystem(psychrolib.SI)


OCCUPANCY_DICT = {
    "OCUNOCCUPIED": 0,
    "UNOCCUPIED": 0,
    False: 0,
    "OCCUPIED": 1,
    "IN$20USE": 1,
    "BYPASS": 1,
    "OCOCCUPIED": 1,
    True: 1,
    "STANDBY": 2,
    "NUL": np.nan,
    "OCNUL": np.nan,
}


def clean_occupancy_df(df, occ_dict=OCCUPANCY_DICT):

    # sort index
    df = df.sort_index()

    # convert to common terms
    df = df.map(lambda x: x.upper() if isinstance(x, str) else x).map(
        lambda x: occ_dict.get(x, np.nan)
    )

    # round to nearest hour
    df.index = (df.index + pd.Timedelta(minutes=30)).floor("h")

    # keep last row per hour, but don't let NaNs overwrite real values
    df = df.ffill().groupby(level=0).last()

    # put on hourly grid
    df = df.asfreq("1h").ffill()

    # for series that start with NA
    df = df.bfill()

    # drop NA if all NA
    df = df.dropna(how="all", axis=1)

    return df


def load_zones(dataset, project, name, clean_data=False, resample_data=False):
    """
    Load zonal data for a particular dataset, building, filter

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str
    project : str
    name : str
    clean_data : bool
    resample_data : bool

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as zones.

    Notes
    -----
    Dependent variables are calculated from core variables

    Examples
    --------
    `datasets.load_zones("2022", "OFF-2", "zone-temps")
    """
    if name in utils.VARIABLE_DEPENDENCIES:
        try:
            data_dict = {}
            variables = utils.VARIABLE_DEPENDENCIES[name]
            for this_var in variables:
                this_df = load_zones(
                    dataset,
                    project,
                    this_var,
                    clean_data=clean_data,
                    resample_data=resample_data,
                )
                data_dict[this_var] = this_df
            fn = utils.FUNCTIONS[name]
            if "project" in inspect.signature(fn).parameters:
                df = fn(project, data_dict)
            else:
                df = fn(data_dict)
            return df
        except Exception:
            print(f"Could not load {name}")
    else:
        filename = DATASETS_PATH / dataset / f"{project}_{name}.csv"
        try:
            this_df = pd.read_csv(filename, parse_dates=True, index_col=0)

            if name == "zone-occupancy_cmd":
                this_df = clean_occupancy_df(this_df)

            if clean_data:
                this_df = clean_df(
                    df=this_df,
                    this_var=name,
                    only_business_hours=False,
                    no_weekends=False,
                    SI_units=False,
                )
            if resample_data and isinstance(this_df.index, pd.DatetimeIndex):
                this_df = this_df.resample("1h").mean()

            if project == "OFF-1" and name == "zone-map":
                # edge case; maps changed
                this_df.index = this_df.index.str.replace(" ", "").str.replace("-", "")
            return this_df
        except FileNotFoundError:
            print(f"Could not find {filename}")


def pull_from_dataset(
    dataset, projects, this_var, clean_data=False, resample_data=False
):
    """
    Helper function to pull from datasets for several buildings

    Parameters
    ----------
    dataset : str
    projects : list
    this_var : str
    clean_data : bool
    resample_data : bool

    Returns
    -------
    dfs in dict form
    """
    dfs = {}
    for project in projects:
        dfs[project] = load_zones(
            dataset=dataset,
            project=project,
            name=this_var,
            clean_data=clean_data,
            resample_data=resample_data,
        )
    return dfs


def load_building(dataset, utility):
    """
    Loads building-level data for a particular dataset, utility

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str
    utility : str, "E" (electricity) or "C" (cooling)

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as buildings.

    Examples
    --------
    `datasets.load_building("2022", "C")`
    """
    if utility not in ["E", "C", "H"]:
        raise ValueError(f"utility: expected one of E,C,H but got {utility}")
    filename = DATASETS_PATH / dataset / f"{utility}.csv"
    try:
        return pd.read_csv(filename, parse_dates=True, index_col=0)
    except FileNotFoundError:
        print(f"Could not find {filename}")


def load_weather(dataset):
    """
    Loads weather data for a particular dataset

    Raises an error if the dataset is unavailable.

    Parameters
    ----------
    dataset : str

    Returns
    -------
    pd.DataFrame
        df with index as timestamps and columns as weather variables.

    Examples
    --------
    `datasets.load_weather("2022")`
    """
    filename = DATASETS_PATH / dataset / "weather.csv"
    try:
        df = pd.read_csv(filename, parse_dates=True, index_col=0)
        df = df[~df.index.duplicated(keep="first")]  # ensure unique index
        df["Enthalpy (kJ/kg)"] = utils.calculate_enthalpy(
            df["temperature"], utils.calculate_W(df["temperature"], df["RH"] / 100)
        )
        return df
    except FileNotFoundError:
        print(f"Could not find {filename}")
