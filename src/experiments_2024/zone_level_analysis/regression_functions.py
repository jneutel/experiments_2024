import numpy as np
import pandas as pd
import statsmodels.api as sm
from experiments_2024 import DATASETS_PATH


BAD_REGRESSION_DATA_2024 = {
    "OFF-2": [],
    "OFF-3": [],
    "OFF-4": [
        # very cold days
        pd.Timestamp("09-16-2024"),
        pd.Timestamp("09-17-2024"),
    ],
    "OFF-5": [],
    "OFF-6": [
        # very cold days
        pd.Timestamp("09-16-2024"),
        pd.Timestamp("09-17-2024"),
    ],
    "OFF-7": [
        # accidentally used a non-useful set of dominant zones
        pd.Timestamp("08-22-2024"),
        # turn over project
        pd.Timestamp("09-05-2024"),
        pd.Timestamp("09-06-2024"),
        pd.Timestamp("09-10-2024"),
        pd.Timestamp("09-11-2024"),
        # very cold days
        pd.Timestamp("09-16-2024"),
        pd.Timestamp("09-17-2024"),
    ],
}

CONTROL_FOR_SUMMER = {
    "OFF-2": True,
    "OFF-3": True,
    "OFF-4": False,
    "OFF-5": False,
    "OFF-6": False,
    "OFF-7": False,  # true if include trial 3
}

FORMAL_TRIALS_2024_START = {
    "OFF-2": pd.Timestamp("06-19-2024"),
    "OFF-3": pd.Timestamp("06-27-2024"),
    "OFF-4": pd.Timestamp("06-19-2024"),
    "OFF-5": pd.Timestamp("06-19-2024"),
    "OFF-6": pd.Timestamp("07-12-2024"),
    "OFF-7": pd.Timestamp("06-19-2024"),
}

FORMAL_TRIALS_2024_END = {
    "OFF-2": pd.Timestamp("09-11-2024"),
    "OFF-3": pd.Timestamp("09-11-2024"),
    "OFF-4": pd.Timestamp("09-18-2024"),
    "OFF-5": pd.Timestamp("09-11-2024"),
    "OFF-6": pd.Timestamp("09-18-2024"),
    "OFF-7": pd.Timestamp("09-18-2024"),
}


def get_2024_binary_df(
    project,
    freq="daily",
    baseline_column="Control",
    drop_baseline_column=True,
    no_weekends=True,
    control_for_weekends=True,
    control_for_summer=False,
    delete_days=[],
    off4_all_zone="adjust",
    off7_trial_3="drop",
):
    """
    Creates binary df to be used in 2024 regression analyses

    Parameters
    ----------
    project : str
        building to create binary df for
    freq : str
        daily, hourly, 15min
    baseline_column : str
        which column to use as baseline
        default is "Control"
    drop_baseline_column : bool
        whether or not to drop baseline column
        default is True
    no_weekends : bool
        whether to exclude weekend data or not
        default is True
    control_for_weekends
        whether to control for weekends or not
        default is True, but ignored if no_weekends is True
    control_for_summer : bool
        whether or not to to control for end of summer
    delete_days : list
        list of days to nan out
    off4_all_zone : str
        how to handle OFF-4 all zone
        "adjust" mean label all zone data prior to Juuly 13 as Poor Control
        otherwise, leave as is
    off7_trial_3 : str
        how to handle OFF-7 trial 3
        "adjust" or "drop"

    Returns
    -------
    df

    Notes
    -----
    # 0 - Control Day
    # 1 - All-Zone
    # 2 - Dominant zones trial 1
    # 3 - Dominant zones trial 2
    # 4 - Dominant zones trial 3
    # 5 - Dominant zones ... trial 1 (bad) in OFF-4, trial 3 redo in OFF-7
    """
    # prep encoded schedule
    encoded_schedule = pd.read_csv(
        f"{DATASETS_PATH}/csvs/2024_experiment_csvs/formal_trials_daily_plan.csv"
    ).set_index("Unnamed: 0")
    encoded_schedule.index.name = None
    encoded_schedule.index = pd.to_datetime(encoded_schedule.index)

    if freq == "hourly":
        encoded_schedule = encoded_schedule.resample("H").ffill()
    elif freq == "15min":
        encoded_schedule = encoded_schedule.resample("15T").ffill()

    encoded_schedule = encoded_schedule[project]
    encoded_schedule.dropna(inplace=True)

    # convert to binary schedule
    trials = ["Control", "All-Zone", "Trial 1", "Trial 2", "Trial 3"]
    binary_df = pd.DataFrame(0, index=list(encoded_schedule.index), columns=trials)
    for i in range(len(trials)):
        binary_df.loc[encoded_schedule == i, trials[i]] = 1

    # optionally control for end of summmer
    if control_for_summer:
        binary_df["End Summer"] = 0
        binary_df.loc[binary_df.index >= pd.Timestamp("08-19-2024"), "End Summer"] = 1

    # building specific adjustments
    if project == "OFF-4":
        binary_df["Trial 1 (Poor Control)"] = 0
        binary_df.loc[encoded_schedule == 5, "Trial 1 (Poor Control)"] = 1
        if off4_all_zone == "adjust":
            binary_df["All-Zone (Poor Control)"] = 0
            for idx in binary_df.index:
                if (
                    idx >= pd.Timestamp("06-19-2024")
                    and idx <= pd.Timestamp("07-13-2024")
                    and binary_df.loc[idx, "All-Zone"] == 1
                ):
                    binary_df.loc[idx, "All-Zone (Poor Control)"] = 1
                    binary_df.loc[idx, "All-Zone"] = 0

    if project == "OFF-7":
        if off7_trial_3 == "adjust":
            binary_df["Trial 3 (Bad)"] = binary_df["Trial 3"]
            binary_df["Trial 3"] = 0
            binary_df.loc[encoded_schedule == 5, "Trial 3"] = 1
        else:
            binary_df = binary_df.loc[binary_df.index <= pd.Timestamp("08-02-2024"), :]
            binary_df = binary_df.drop(columns=["Trial 3"])

    # handle baseline
    if drop_baseline_column:
        # the baseline is represented in absentia
        binary_df.drop(columns=baseline_column, inplace=True)
    # handle weekends
    if no_weekends:
        binary_df.loc[binary_df.index.dayofweek >= 5, :] = np.nan
    if (not no_weekends) and control_for_weekends:
        binary_df["Weekend"] = 0
        binary_df.loc[binary_df.index.dayofweek >= 5, "Weekend"] = 1
    # handle bad data
    for idx in binary_df.index:
        if idx.floor("D") in delete_days:
            binary_df.loc[idx, :] = np.nan
    binary_df.dropna(how="any", inplace=True)
    return binary_df


def general_Delta_fn(
    df, T, binary, mode="Absolute Change", summary_statistic="Mean", return_cov=False
):
    """
    General purpose function to find delta due to setpoint change

    Parameters
    ----------
    df : pd.DataFrame
        df with time as index (hourly) and equips as cols
    T : pd.Series
        outside temperature data of interest (hourly)
    binary : pd.Series or pd.DataFrame
        0 if not test day, 1 if test day
        each column represents a different test, e.g. all zone day vs dominant zone day
    mode : str
        "Absolute Change" or "Percent Change"
    summary_statistic : str
        "Mean", "Sum", "Max", "Min"
    return_cov : bool
        whether to return covariance matrices in dict

    Returns
    -------
    pd.DataFrame, results of regression
    """
    # allow for pd.DataFrame input as T
    if T is not None and isinstance(T, pd.DataFrame):
        T = T["temperature"]

    # allow for pd.Series input as binary
    if isinstance(binary, pd.Series):
        binary = binary.to_frame()
        binary.columns = ["High SP"]

    # prep raw data
    if summary_statistic == "Mean":
        df = df.groupby(df.index.date).mean()
    if summary_statistic == "Sum":
        df = df.groupby(df.index.date).sum(min_count=1)
    if summary_statistic == "Max":
        df = df.groupby(df.index.date).max()
    if summary_statistic == "Min":
        df = df.groupby(df.index.date).min()
    df.index = pd.DatetimeIndex(df.index)
    if mode == "Percent Change":
        df[df <= 0] = np.nan  # causes error with log

    binary.dropna(how="all", inplace=True)
    binary.index = pd.DatetimeIndex(binary.index)
    if T is not None:
        T = T.groupby(T.index.date).mean()
        T.index = pd.DatetimeIndex(T.index)
        common = binary.index.intersection(df.index).intersection(T.index, sort=True)
        binary = binary.loc[common, :]
        df = df.loc[common, :]
        T = T.loc[common]
        vars = ["OAT", "Intercept"]
    else:
        common = binary.index.intersection(df.index, sort=True)
        binary = binary.loc[common, :]
        df = df.loc[common, :]
        vars = ["Intercept"]
    # initializations
    equips = list(df.columns)
    tests = list(binary.columns)
    vars.extend(tests)
    cols = []
    for var in vars:
        cols.append(f"Slope {var}")
        cols.append(f"Slope Low {var}")
        cols.append(f"Slope High {var}")
        cols.append(f"Std Err {var}")
        cols.append(f"P-Value {var}")
    for test in tests:
        cols.append(f"Delta {test}")
        cols.append(f"Delta Low {test}")
        cols.append(f"Delta High {test}")
    cols.append("R2")
    all_results = pd.DataFrame(data=np.nan, index=equips, columns=cols)
    all_covs = {}
    # run regression for each equip
    for equip in equips:
        # filter data for this equip
        ys = df[equip]
        ys = ys.dropna()
        if len(ys) < 3:
            all_results.loc[equip, :] = np.nan
            continue
        if mode == "Percent Change":
            ys = np.log(ys)
        intercept = pd.Series(1, index=ys.index)
        bins = binary.loc[ys.index, :]
        if T is not None:
            ts = T.loc[ys.index]
            xs = pd.concat([ts, intercept, bins], axis=1)
        else:
            xs = pd.concat([intercept, bins], axis=1)
        xs.columns = vars

        # fit model and store info
        model = sm.OLS(ys, xs)
        results = model.fit()
        for var in vars:
            all_results.loc[equip, f"Slope {var}"] = results.params[var]
            all_results.loc[equip, f"Std Err {var}"] = results.bse[var]
            all_results.loc[equip, f"Slope Low {var}"] = (
                results.params[var] - 1.96 * results.bse[var]
            )
            all_results.loc[equip, f"Slope High {var}"] = (
                results.params[var] + 1.96 * results.bse[var]
            )
            all_results.loc[equip, f"P-Value {var}"] = results.pvalues[var]
        all_results.loc[equip, "R2"] = results.rsquared
        all_covs[equip] = results.cov_params()

    if mode == "Percent Change":
        for test in tests:
            for err in [" ", " Low ", " High "]:
                all_results[f"Delta{err}{test}"] = 100 * (
                    np.exp(all_results[f"Slope{err}{test}"].astype(float)) - 1
                )
    else:
        for test in tests:
            for err in [" ", " Low ", " High "]:
                all_results[f"Delta{err}{test}"] = all_results[f"Slope{err}{test}"]

    if return_cov:
        return (all_results, all_covs)
    return all_results


def general_regression_fn(y_data, x_data, summary_statistic="Mean"):
    """
    General purpose function to regress y onto x by equip

    Parameters
    ----------
    y_data : pd.DataFrame
        y_data with time as index (hourly) and equips as cols
    x_data : pd.Series
        x_data with time as index (hourly) and equips as cols
    summary_statistic : str
        "Mean", "Sum", or "Max"

    Returns
    -------
    pd.DataFrame, results of regression
    """
    # prep raw data
    if summary_statistic == "Mean":
        y_data = y_data.groupby(y_data.index.date).mean()
    if summary_statistic == "Sum":
        y_data = y_data.groupby(y_data.index.date).sum(min_count=1)
    if summary_statistic == "Max":
        y_data = y_data.groupby(y_data.index.date).max()

    y_data.index = pd.DatetimeIndex(y_data.index)
    x_data = x_data.groupby(x_data.index.date).mean()
    x_data.index = pd.DatetimeIndex(x_data.index)
    common = y_data.index.intersection(x_data.index, sort=True)
    y_data = y_data.loc[common, :]
    x_data = x_data.loc[common, :]
    # initializations
    equips = list(y_data.columns)
    all_results = pd.DataFrame(
        index=equips,
        columns=[
            "Slope X",
            "Slope Intercept",
            "Std Err X",
            "Std Err Intercept",
            "P-Value X",
            "P-Value Intercept",
            "R2",
        ],
    )
    # run regression for each equip
    for equip in equips:
        # filter data for this equip
        ys = y_data[equip]
        ys = ys.dropna()
        xs = x_data[equip]
        xs = xs.dropna()
        common = ys.index.intersection(xs.index, sort=True)
        ys = ys[common]
        xs = xs[common]
        if len(ys) < 3:
            all_results.loc[equip, :] = np.nan
            continue
        xs = xs.to_frame()
        xs["Intercept"] = pd.Series(1, index=xs.index)
        vars = ["X", "Intercept"]
        xs.columns = vars
        # fit model and store info
        model = sm.OLS(ys, xs)
        results = model.fit()
        for var in vars:
            all_results.loc[equip, f"Slope {var}"] = results.params[var]
            all_results.loc[equip, f"Std Err {var}"] = results.bse[var]
            all_results.loc[equip, f"P-Value {var}"] = results.pvalues[var]
        all_results.loc[equip, "R2"] = results.rsquared
    return all_results
