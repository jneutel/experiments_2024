import numpy as np
import pandas as pd
import psychrolib
import warnings

from experiments_2024 import constants
from experiments_2024.zone_level_analysis.cleaning import clean_columns
from experiments_2024.zone_level_analysis.base import (
    trim_to_common_elements,
    run_vav_to_ahu,
)

psychrolib.SetUnitSystem(psychrolib.SI)
warnings.filterwarnings("ignore")


AVAILABLE_PROJECTS = [
    "OFF-2",
    "OFF-3",
    "OFF-4",
    "OFF-5",
    "OFF-6",
    "OFF-7",
]

AVAILABLE_VARIABLES = [
    "ahu-chwv_cmd",  # ahu core
    "ahu-dap",
    "ahu-dapsp",
    "ahu-fanspeed",
    "ahu-econ_cmd",
    "ahu-oa_damper",
    "ahu-ra_damper_cmd",
    "ahu-dat",
    "ahu-datsp",
    "ahu-rat",
    "ahu-mat",
    "ahu-oat",
    "ahu-power",
    "ahu-frequency",
    "zone-occupancy_eff",  # zone core
    "zone-occupancy_cmd",
    "zone-dat",
    "zone-datsp",
    "zone-airflow",
    "zone-airflowsp",
    "zone-coolsp",
    "zone-heatsp",
    "zone-temps",
    "zone-tloads",
    "zone-damper",
    "zone-rhv",
    "zone-zonesp",
    "zone-cool_offset",
    "zone-heat_offset",
    "zone-map",  # zone derived
    "zone-dat_ahu",
    "zone-deadband_top",
    "zone-deadband_bottom",
    "zone-local_offset",
    "zone-deviation_coolsp",
    "zone-deviation_heatsp",
    "zone-temps_norm",
    "zone-deviation_dat_datsp",
    "zone-deviation_dat_datahu",
    "zone-deviation_airflow",
    "zone-norm_deviation_airflow",
    "zone-simple_cooling_requests",
    "zone-simple_pressure_requests",
    "ahu-simple_cooling_requests",  # ahu derived
    "ahu-airflow",
    "ahu-airflowsp",
]

VARIABLE_DEPENDENCIES = {
    "zone-dat_ahu": ["ahu-dat", "zone-map"],
    "zone-deadband_top": ["zone-zonesp", "zone-cool_offset"],
    "zone-deadband_bottom": ["zone-zonesp", "zone-heat_offset"],
    "zone-local_offset": ["zone-zonesp", "zone-cool_offset", "zone-coolsp"],
    "zone-deviation_coolsp": ["zone-temps", "zone-coolsp"],
    "zone-deviation_heatsp": ["zone-temps", "zone-heatsp"],
    "zone-temps_norm": ["zone-temps", "zone-coolsp", "zone-heatsp"],
    "zone-deviation_dat_datsp": ["zone-dat", "zone-datsp"],
    "zone-deviation_dat_datahu": ["zone-dat", "ahu-dat", "zone-map"],
    "zone-deviation_airflow": ["zone-airflow", "zone-airflowsp"],
    "zone-norm_deviation_airflow": ["zone-airflow", "zone-airflowsp"],
    "zone-simple_cooling_requests": ["zone-tloads"],
    "zone-simple_pressure_requests": ["zone-damper"],
    "ahu-simple_cooling_requests": ["zone-map", "zone-tloads"],
    "ahu-airflow": ["zone-map", "zone-airflow"],
    "ahu-airflowsp": ["zone-map", "zone-airflowsp"],
}


# HELPER FUNCTIONS


def get_moist_air_density(T, W):
    """
    Computes the density of moist air (kg/m3) as function of T

    Parameters
    ----------
    T : pd.Series or pd.DataFrame
        temp in F
    W : pd.Series or pd.DataFrame
        humidity ratio

    Returns
    -------
    pandas.Series or pd.DataFrame

    Notes
    -----
    https://www.engineeringtoolbox.com/density-air-d_680.html see eq 3
    """
    T_c = constants.F_to_C(T)
    T_k = T_c + 273.15  # degrees K
    P = 101325  # atmospheric pressure (Pa or J/m3)
    Ra = 286.9  # gas constant dry air (J/kgK)
    Rw = 461.5  # gas constant water vapor (J/kgK)
    moist_density = ((P / (Ra * T_k)) * (1 + W)) / (1 + W * (Rw / Ra))
    return moist_density


def calculate_W(T, RH):
    """
    Calculates humidity ratio - kg water / kg dry air

    Parameters
    ----------
    T : pd.Series
        temperature in F
    RH : pd.Series
        relative humidity in decimal [0, 1]

    Returns
    -------
    pd.Series
    """
    idx = list(T.index)
    T_c = constants.F_to_C(T)  # degrees C

    # calculate humidity ratio
    humidity_ratio = pd.Series(index=idx)  # kg H2O / kg dry air
    for i in idx:
        humidity_ratio[i] = psychrolib.GetHumRatioFromRelHum(
            TDryBulb=T_c[i],  # celsius
            RelHum=RH[i],  # [0, 1]
            Pressure=101325,  # pascal
        )
    return humidity_ratio


def calculate_enthalpy(T, W):
    """
    Calculates enthalpy in kJ/kg

    Parameters
    ----------
    T : pd.Series
        temperature in F
    W : pd.Series
        humidity ratio

    Returns
    -------
    pd.Series
    """
    idx = list(T.index)
    T_c = constants.F_to_C(T)  # degrees C
    enthalpy_of_moist_air = pd.Series(index=idx)  # kJ / kg
    for i in idx:
        enthalpy_of_moist_air[i] = (
            psychrolib.GetMoistAirEnthalpy(
                TDryBulb=T_c[i], HumRatio=W[i]  # celsius,  # kg H2O / kg dry air
            )
            / 1000
        )  # from J to kJ

    return enthalpy_of_moist_air


# TEMPS


def compute_zone_dat_ahu(data_dict):
    """
    Compute zone-dat at the ahu level, for each vav.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"ahu-dat": df1, "zone-map": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    dat = data_dict["ahu-dat"]
    vav_to_ahu = data_dict["zone-map"]
    dat_ahu = pd.DataFrame(index=dat.index)
    for ahu in dat.columns:
        if ahu in vav_to_ahu.values:
            zones = list(vav_to_ahu[vav_to_ahu == ahu].dropna().index)
            for zone in zones:
                dat_ahu[zone] = dat[ahu]
    return dat_ahu


def compute_zone_deadband_top(project, data_dict):
    """
    Computes the top of the deadband without local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-cool_offset": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-cool_offset
    - Returns data in F.
    """
    # havas does not have cooling offset
    if project in ["HAVAS", "OFF-3"]:
        return data_dict["zone-zonesp"]
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    middle = data_dict["zone-zonesp"]
    offset = data_dict["zone-cool_offset"]
    middle, offset = trim_to_common_elements([middle, offset])
    return middle + offset.abs()


def compute_zone_deadband_bottom(project, data_dict):
    """
    Computes the bottom of the deadband without local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-heat_offset": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-heat_offset
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    middle = data_dict["zone-zonesp"]
    offset = data_dict["zone-heat_offset"]
    middle, offset = trim_to_common_elements([middle, offset])
    return middle - offset.abs()


def compute_zone_local_offset(project, data_dict):
    """
    Computes the local offset

    Parameters
    ----------
    project : str
        name of project
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-zonesp": df1, "zone-cool_offset": df2, "zone-coolsp" : df3}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-zonesp, zone-cool_offset, zone-coolsp
    - Returns data in F.
    """
    for this_var in ["zone-coolsp"]:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    eff_coolsp = data_dict["zone-coolsp"]
    top_deadband = compute_zone_deadband_top(
        project,
        {
            "zone-zonesp": data_dict["zone-zonesp"],
            "zone-cool_offset": data_dict["zone-cool_offset"],
        },
    )
    eff_coolsp, top_deadband = trim_to_common_elements([eff_coolsp, top_deadband])
    return eff_coolsp - top_deadband


def compute_zone_deviation_coolsp(data_dict):
    """
    Computes the deviation between zone temperatures and zone cooling setpoints.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-temps": df1, "zone-coolsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-temps, zone-coolsp.
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    temp = data_dict["zone-temps"]
    coolsp = data_dict["zone-coolsp"]
    temp, coolsp = trim_to_common_elements([temp, coolsp])
    return temp - coolsp


def compute_zone_deviation_heatsp(data_dict):
    """
    Computes the deviation between zone temperatures and zone heating setpoints.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-temps": df1, "zone-heatsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-temps, zone-heatsp.
    - Returns data in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    temp = data_dict["zone-temps"]
    heatsp = data_dict["zone-heatsp"]
    temp, heatsp = trim_to_common_elements([temp, heatsp])
    return temp - heatsp


def compute_zone_temps_norm(data_dict):
    """
    Computes the normalized temperture

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-temps": df1, "zone-coolsp": df2, "zone-heatsp": df3}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-temps, zone-coolsp, zone-heatsp
    - Returns data in unitless.
    """
    temp = data_dict["zone-temps"]
    coolsp = data_dict["zone-coolsp"]
    heatsp = data_dict["zone-heatsp"]
    temp, coolsp, heatsp = trim_to_common_elements([temp, coolsp, heatsp])
    norm = (temp - heatsp) / (coolsp - heatsp)
    norm[(coolsp - heatsp) == 0] = np.nan  # scrub inf
    return norm


def compute_zone_deviation_dat_datsp(data_dict):
    """
    Compute the deviation of zone-dat from zone-datsp.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-dat": df1, "zone-datsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-dat, zone-datsp
    - Returns in F
    """
    dat = data_dict["zone-dat"]
    datsp = data_dict["zone-datsp"]
    dat, datsp = trim_to_common_elements([dat, datsp])
    return dat - datsp


def compute_zone_deviation_dat_datahu(data_dict):
    """
    Computes temp difference from outlet of AHU to outlet of zone

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-dat": df1, "ahu-dat": df2, "zone-map": df3}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-dat, dat, zone-map
    - Return in F.
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    zone_dat = data_dict["zone-dat"]
    ahu_dat = data_dict["ahu-dat"]
    zone_map = data_dict["zone-map"]
    zone_dat_ahu = compute_zone_dat_ahu({"ahu-dat": ahu_dat, "zone-map": zone_map})
    zone_dat, zone_dat_ahu = trim_to_common_elements([zone_dat, zone_dat_ahu])
    return zone_dat - zone_dat_ahu


# AIRFLOW


def compute_zone_deviation_airflow(data_dict):
    """
    Compute the deviation of zone airflow from the setpoint.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-airflow": df1, "zone-airflowsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-airflow, zone-airflowsp
    - Returns in cfm
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    airflow = data_dict["zone-airflow"]
    airflowsp = data_dict["zone-airflowsp"]
    airflow, airflowsp = trim_to_common_elements([airflow, airflowsp])
    return airflow - airflowsp


def compute_zone_norm_deviation_airflow(data_dict):
    """
    Compute airflow / airflowsp.

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-airflow": df1, "zone-airflowsp": df2}
        df: index time, column is equip

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-airflow, zone-airflowsp
    - Returns in %
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    airflow = data_dict["zone-airflow"]
    airflowsp = data_dict["zone-airflowsp"]
    airflow, airflowsp = trim_to_common_elements([airflow, airflowsp])
    norm_dev = airflow / airflowsp
    norm_dev[airflowsp == 0] = np.nan  # scrub inf
    return norm_dev


# REQUESTS


def compute_zone_simple_cooling_requests(data_dict):
    """
    1 if tload above 70%, 0 if not

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-tloads", df1}

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-tloads
    - Unitless
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    tload = data_dict["zone-tloads"]
    df = pd.DataFrame(0, index=tload.index, columns=tload.columns)
    df[tload >= 70] = 1
    return df


def compute_zone_simple_pressure_requests(data_dict):
    """
    1 if damper above 70%, 0 if not

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case, {"zone-damper", df1}

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    - Assumes that we can load zone-damper
    - Unitless
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    damper = data_dict["zone-damper"]
    df = pd.DataFrame(0, index=damper.index, columns=damper.columns)
    df[damper >= 70] = 1
    return df


# AHU vars


def compute_ahu_cooling_requests(data_dict):
    """
    Calculates total estimated unweighted cooling requests bottom up using tload

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case ...
        {
            "zone-map" : df1,
            "zone-tloads" : df2,
        }

    Returns
    -------
    ahu-simple_cooling_requests (unitless)
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    zone_map = data_dict["zone-map"]
    zone_tload = data_dict["zone-tloads"]
    zone_CRs = compute_zone_simple_cooling_requests({"zone-tloads": zone_tload})
    common = list(zone_map.index.intersection(zone_CRs.columns))
    zone_map = zone_map.loc[common, :]
    zone_CRs = zone_CRs.loc[:, common]
    ahus = list(set(zone_map["AHU"]))
    summed_CRs = pd.DataFrame(index=zone_CRs.index, columns=ahus)
    for ahu in ahus:
        these_zones = list(((zone_map[zone_map["AHU"] == ahu]).dropna()).index)
        this_CR = zone_CRs.loc[:, these_zones]
        summed_CRs[ahu] = this_CR.sum(axis=1)
    summed_CRs = clean_columns(summed_CRs, "ahu-dummy")
    return summed_CRs


def compute_ahu_airflow(data_dict):
    """
    Calculates total airflow supply bottom up using zone airflow

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case ...
        {
            "zone-map" : df1,
            "zone-airflow" : df2,
        }

    Returns
    -------
    ahu-airflow in units of CFM
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    zone_map = data_dict["zone-map"]
    zone_airflow = data_dict["zone-airflow"]
    common = list(zone_map.index.intersection(zone_airflow.columns))
    zone_map = zone_map.loc[common, :]
    zone_airflow = zone_airflow.loc[:, common]
    summed_supply_air = run_vav_to_ahu(
        df=zone_airflow, map=zone_map["AHU"], vav_to_ahu="Sum"
    )
    summed_supply_air = clean_columns(summed_supply_air, "ahu-airflow")
    return summed_supply_air


def compute_ahu_airflowsp(data_dict):
    """
    Calculates total airflow supply bottom up using zone airflow

    Parameters
    ----------
    data_dict : dict
        dictionary containing the data needed for this calc
        keys of dictionary are variable name
        value is the df
        in this case ...
        {
            "zone-map" : df1,
            "zone-airflowsp" : df2,
        }

    Returns
    -------
    ahu-airflowsp in units of CFM
    """
    for this_var in data_dict:
        data_dict[this_var] = clean_columns(data_dict[this_var], this_var)
    zone_map = data_dict["zone-map"]
    zone_airflowsp = data_dict["zone-airflowsp"]
    common = list(zone_map.index.intersection(zone_airflowsp.columns))
    zone_map = zone_map.loc[common, :]
    zone_airflowsp = zone_airflowsp.loc[:, common]
    summed_supply_airsp = run_vav_to_ahu(
        df=zone_airflowsp, map=zone_map["AHU"], vav_to_ahu="Sum"
    )
    summed_supply_airsp = clean_columns(summed_supply_airsp, "ahu-airflow")
    return summed_supply_airsp


# vars --> functions

FUNCTIONS = {
    "zone-dat_ahu": compute_zone_dat_ahu,
    "zone-deadband_top": compute_zone_deadband_top,
    "zone-deadband_bottom": compute_zone_deadband_bottom,
    "zone-local_offset": compute_zone_local_offset,
    "zone-deviation_coolsp": compute_zone_deviation_coolsp,
    "zone-deviation_heatsp": compute_zone_deviation_heatsp,
    "zone-temps_norm": compute_zone_temps_norm,
    "zone-deviation_dat_datsp": compute_zone_deviation_dat_datsp,
    "zone-deviation_dat_datahu": compute_zone_deviation_dat_datahu,
    "zone-deviation_airflow": compute_zone_deviation_airflow,
    "zone-norm_deviation_airflow": compute_zone_norm_deviation_airflow,
    "zone-simple_cooling_requests": compute_zone_simple_cooling_requests,
    "zone-simple_pressure_requests": compute_zone_simple_pressure_requests,
    "ahu-simple_cooling_requests": compute_ahu_cooling_requests,
    "ahu-airflow": compute_ahu_airflow,
    "ahu-airflowsp": compute_ahu_airflowsp,
}
