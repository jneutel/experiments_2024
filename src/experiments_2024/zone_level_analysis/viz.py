# math
import copy
import functools
import math
import numpy as np
import pandas as pd
import re

# plotting
import plotly.graph_objects as go
import plotly.subplots as sbplt
from plotly.subplots import make_subplots
from matplotlib import colors as mcolors

# buildings
from experiments_2024.zone_level_analysis.cleaning import OCCUPANCY_TIME_RANGE
from experiments_2024.zone_level_analysis.base import trim_to_common_elements

# FORMATTING
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 500

COLORS = {
    0: "Black",
    1: "MediumBlue",
    2: "DarkOrange",
    3: "Firebrick",
    4: "ForestGreen",
    5: "Orchid",
    6: "SlateBlue",
    7: "Coral",
    8: "Teal",
    9: "Goldenrod",
    10: "MediumVioletRed",
}
SHAPES = {0: "circle", 1: "x", 2: "triangle-up", 3: "square", 4: "cross", 5: "asterisk"}
TITLE_SZ = 30  # text size
TXT_SZ = 22  # text size
LEGEND_SZ = 26
MRKR_SZ = 8  # marker size
GRAPH_NUM_COLS = 3
HORIZONTAL_SPACING = 0.1
VERTICAL_SPACING = None


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def force_dict(obj, title):
    if (obj is not None) and (not isinstance(obj, dict)):
        obj = {title: obj}
    return obj


def get_err_values(ser_pos, ser_neg, zone):
    if ser_pos is not None:
        pos_val = [ser_pos[zone]]
    else:
        if ser_neg is None:
            pos_val = None
        else:
            pos_val = [0]

    if ser_neg is not None:
        neg_val = [ser_neg[zone]]
    else:
        if ser_pos is None:
            neg_val = None
        else:
            neg_val = [0]

    return pos_val, neg_val


def adjust_axis_bound(bound_type, bound, multiplier=0.05):
    """
    Used for axis formatting - increments bound slightly above min/max point

    Parameters
    ----------
    bound_type : str
        "min" or "max"
    bound : float
        the original min or max value
    multiplier : float
        % to increment beyond min/max point
        default = 0.1

    Returns
    -------
    The new min/max value bound (float)
    """
    if bound_type == "min":
        if bound < 0:
            bound = bound * (1 + multiplier)
        else:
            bound = bound * (1 - multiplier)
    else:
        if bound < 0:
            bound = bound * (1 - multiplier)
        else:
            bound = bound * (1 + multiplier)
    return bound


def find_min_max(this_dict, these_keys=None):
    """
    Find global min and max value of dict(dfs)

    Parameters
    ----------
    this_dict : dict(pd.DataFrame)
    these_keys : list(str)

    Returns
    -------
    (min, max)
    """
    min_value = float("inf")
    max_value = float("-inf")
    if these_keys is None:
        these_keys = list(this_dict.keys())
    for this_key in these_keys:
        this_df = this_dict[this_key]
        if this_df.min(axis=None) < min_value:
            min_value = this_df.min(axis=None)
        if this_df.max(axis=None) > max_value:
            max_value = this_df.max(axis=None)
    return (min_value, max_value)


def get_background_shading(idx, shade_weekends=False, sps=None):
    """
    Get background shading for time series, auto shades out non-business hours and weekends

    Parameters
    ----------
    idx : pandas index
        time series that we would like to add shading for
    shade_weekends : bool
    sps : pd.Series
        optional sp schedule

    Returns
    -------
    df with shading instructions
    """
    shading_full = pd.DataFrame(np.nan, index=idx, columns=["Day Type", "Color"])
    # sp colors if applicable
    C = {}
    if sps is not None:
        unique = sps.unique()
        for i in range(len(unique)):
            C[unique[i]] = COLORS[i]
    for this_idx in idx:
        if (
            this_idx.hour < OCCUPANCY_TIME_RANGE[0].hour
            or this_idx.hour >= OCCUPANCY_TIME_RANGE[1].hour
        ):
            # after hours
            shading_full.loc[this_idx, "Day Type"] = "Non-Business Hour"
            shading_full.loc[this_idx, "Color"] = "Grey"
        elif shade_weekends and this_idx.dayofweek >= 5:
            # weekends
            shading_full.loc[this_idx, "Day Type"] = "Non-Business Hour"
            shading_full.loc[this_idx, "Color"] = "Grey"
        else:
            # business hours
            if (sps is not None) and (this_idx in list(sps.index)):
                c = C[sps[this_idx]]
                m = sps[this_idx]
            else:
                c = "White"
                m = "Business Hour"
            shading_full.loc[this_idx, "Color"] = c
            shading_full.loc[this_idx, "Day Type"] = m
    # compress
    shading = pd.DataFrame(np.nan, index=idx, columns=["Day Type", "Color"])
    shading.iloc[0, :] = shading_full.iloc[0, :]
    for i in range(1, len(shading_full)):
        last_idx = shading_full.index[i - 1]
        this_idx = shading_full.index[i]
        if shading_full.loc[last_idx, "Color"] != shading_full.loc[this_idx, "Color"]:
            shading.loc[this_idx, "Color"] = shading_full.loc[this_idx, "Color"]
            shading.loc[this_idx, "Day Type"] = shading_full.loc[this_idx, "Day Type"]
    shading.dropna(inplace=True)
    return shading


def update_fig_formatting(
    fig,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
    showlegend=True,
    title=None,
    x_axis_title="x axis",
    y_axis_title="y axis",
    x_zerolinecolor="LightGray",
    y_zerolinecolor="LightGray",
    grid_color="LightGray",
    x_rangemode="normal",
    y_rangemode="normal",
    x_range=None,
    y_range=None,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
):
    """
    General purpose function that handles figure formatting

    Parameters
    ----------
    fig : plotly figure object
        original figure
    width : float
        width of new figure, float
        default = GRAPH_WIDTH
    height : float
        height of new figure, float
        default = GRAPH_HEIGHT
    showlegend : bool
        whether to include a legend, boolen
        default = True
    title : str
        title of plot
        default = None
    x_axis_title : str
        title of x axis
        default = "x axis"
        special key word "Dont Update" means leave as is
    y_axis_title : str
        title of y axis
        default = "y axis"
        special key word "Dont Update" means leave as is
    x_zerolinecolor : str
        color of y axis (x zero line)
        default = "LightGray"
    y_zerolinecolor : str
        color or x axis (y zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    x_rangemode : str
        how to handle x axis range
        default = "normal"
        "tozero" common alternative
    y_rangemode : str
        how to handle y axis range
        default = "normal"
        "tozero" common alternative
    x_range : list
        list of len(2) [min, max]
        bounds of x axis
        default = None (auto calculated)
    y_range : list
        list of len(2) [min, max]
        bounds of y axis
        default = None (auto calculated)
    title_size : int
        text size of title
        default = TITLE_SZ
        special key word None means dont update
    text_size : int
        text size of axes
        default = TXT_SZ
        special key word None means dont update
    legend_size : int
        text size of legend
        default = TXT_SZ
        special key word None means dont update

    Returns
    -------
    Figure with updated formatting
    """
    if text_size is not None:
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=showlegend,
            font=dict(size=text_size),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
    else:
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=showlegend,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    if x_axis_title == "Dont Update":
        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            gridcolor=grid_color,
            zerolinecolor=x_zerolinecolor,
            rangemode=x_rangemode,
        )
    else:
        fig.update_xaxes(
            title_text=x_axis_title,
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            gridcolor=grid_color,
            zerolinecolor=x_zerolinecolor,
            rangemode=x_rangemode,
        )
    if y_axis_title == "Dont Update":
        fig.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            gridcolor=grid_color,
            zerolinecolor=y_zerolinecolor,
            rangemode=y_rangemode,
        )
    else:
        fig.update_yaxes(
            title_text=y_axis_title,
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            gridcolor=grid_color,
            zerolinecolor=y_zerolinecolor,
            rangemode=y_rangemode,
        )

    # replace overall title
    if title == "Title" or title is None:
        fig.update_layout(title=None, margin=dict(t=40))

    if title_size is not None:
        fig.update_annotations(font_size=title_size)

    if legend_size is not None:
        fig.update_layout(legend=dict(font=dict(size=legend_size)))
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    return fig


def combine_figs(
    figs,
    y_axis_title="Y Label",
    x_axis_title="X Label",
    y_range=None,
    x_range=None,
    force_same_yaxes=True,
    force_same_xaxes=True,
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    Function to convert dictionary of figs into one fig with subplots

    Parameters
    -----------
    figs : dict
        A dictionary where keys are subplot titles (strings) and values are Plotly figures.
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default = "X Label"
    y_range : list
        [min, max]
    x_range : list
        [min, max]
    force_same_yaxes : bool
        whether all subplots have same yaxes
        default = True
    force_same_xaxes : bool
        whether all subplots have same xaxes
        default = True
    num_cols : int
        Number of columns in the subplot grid.
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Figure with updated formatting
    """
    # Extract keys and figures from the dictionary
    keys = list(figs.keys())
    figs = list(figs.values())

    # Calculate number of rows needed based on number of figures and columns
    num_figs = len(figs)
    num_rows = (
        num_figs + num_cols - 1
    ) // num_cols  # Ceiling division to determine rows

    # Create a subplot figure with specified rows and columns
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=keys,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # Add each figure as a subplot with corresponding title
    min_y_value, max_y_value = (10**6, -(10**6))
    min_x_value, max_x_value = (10**6, -(10**6))

    for i, (key, fig_to_add) in enumerate(zip(keys, figs), start=1):
        row = (i - 1) // num_cols + 1
        col = (i - 1) % num_cols + 1
        for trace in fig_to_add.data:
            # min and max for y and x
            if trace.y[0] < min_y_value:
                min_y_value = trace.y[0]
            if trace.y[0] > max_y_value:
                max_y_value = trace.y[0]
            if trace.x[0] < min_x_value:
                min_x_value = trace.x[0]
            if trace.x[0] > max_x_value:
                max_x_value = trace.x[0]
            if i == 1:
                fig.add_trace(trace, row=row, col=col)
            else:
                trace.update(showlegend=False)
                fig.add_trace(trace, row=row, col=col)

    if y_range is None and force_same_yaxes:
        y_range = [
            adjust_axis_bound("min", min_y_value),
            adjust_axis_bound("max", max_y_value),
        ]
    if x_range is None and force_same_xaxes:
        x_range = [
            adjust_axis_bound("min", min_x_value),
            adjust_axis_bound("max", max_x_value),
        ]

    fig = update_fig_formatting(
        fig,
        y_axis_title=y_axis_title,
        x_axis_title=x_axis_title,
        width=width * num_cols,
        height=height * num_rows,
        y_range=y_range,
        x_range=x_range,
    )
    return fig


def make_dot_plot(
    y_data,
    y_error_up_data=None,
    y_error_down_data=None,
    sort_by=None,
    ascending=True,
    normalize_x=False,
    color_data=None,
    shape_data=None,
    opacity_data=None,
    size_data=None,
    color_and_shape=False,
    color_legend=None,
    shape_legend=None,
    color_legend_order=None,
    shape_legend_order=None,
    dont_add_to_legend=[],
    color_override=False,
    labels=None,
    title="Title",
    y_axis_title="Y Label",
    x_axis_title="Zones",
    force_same_yaxes=True,
    y_range=None,
    x_range=None,
    y_zerolinecolor="Black",
    x_zerolinecolor="LightGray",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    marker_size=MRKR_SZ,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    General purpose dot plot function

    Parameters
    ----------
    y_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        each df should have the same columns
        key is the subplot title
        columns are the variables plotted on each plot
    y_error_up_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        optional to add error bars in up direction
        distance from central estimate
        must have same number of keys and columns as y_data
    y_error_down_data : dict(pd.DataFrame) or pd.DataFrame
        {key: pd.DataFrame}
        optional to add error bars in down direction
        distance from central estimate
        must have same number of keys and columns as y_data
    sort_by : str
        which col to sort by
        default = None --> first col
        special str "index" means sort by index alphabetically
        special str "all" means sort each column separately
        special str "dont" means dont sort at all (same order as input)
    ascending : bool
        determines sort order, default True (ascending)
    normalize_x : bool
        if True, x displayed as percentage rather than count of zones
    color_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to color zones by data
        values should start at 0
        dfs should have 1 column (typical), or must have same number of columns as y_data
    shape_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to shape zones by data
        values should start at 0
        dfs should have 1 column, or must have same number of columns as y_data
    opacity_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to opacity zones by data
        dfs should have 1 column, or must have same number of columns as y_data
    size_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to change marker size by zone
        dfs should have 1 column
    color_and_shape : bool
        if True, use the same data for color and shape
        and combine in the legend
    color_legend : dict
        example:
        {"name" : {key1 : "Dominant", key2 : "Dominated"},
         "color" : {key1 : "blue", key2 : "red"}}
    shape_legend : dict
        example:
        {"name" : {key1 : "Included", key2 : "Excluded"},
         "shape" : {key1 : "circle", key2 : "x"}}
    color_legend_order : list
        list of ints determing order to add to legend
        default None, in ascending order
    shape_legend_order : list
        list of ints determing order to add to legend
        default None, in ascending order
    dont_add_to_legend : list
        list of vars not to add to legend
    color_override : bool
        if True, then color_data overrides column coloring even if >1 columns
    labels : dict
        {key : list}
        for each key list of labels to show
    title : str
        only used if plotting one building
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default = "Zones"
    force_same_yaxes : bool
        whether all subplots have same y axis
        default = True
    y_range : list
        if used, we force y axis range to [min, max]
    x_range : list
        if used, we force x axis range to [min, max]
    y_zerolinecolor : str
        color or x axis (y zero line)
        default = "Black"
    x_zerolinecolor : str
        color of y axis (x zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    marker_size : int
        size of markers
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Dot plot figure

    Notes
    -----
    Works with df or dict(dfs)
    df gives you one plot, dict(df) gives you multiple subplots
    """
    if color_and_shape:
        shape_data = color_data

    # force into nested
    y_data = force_dict(y_data, title)
    y_error_up_data = force_dict(y_error_up_data, title)
    y_error_down_data = force_dict(y_error_down_data, title)
    color_data = force_dict(color_data, title)
    opacity_data = force_dict(opacity_data, title)
    shape_data = force_dict(shape_data, title)
    size_data = force_dict(size_data, title)
    labels = force_dict(labels, title)

    # define cols
    these_keys = list(y_data.keys())
    y_cols = list(y_data[these_keys[0]].columns)
    num_y_cols = len(y_cols)

    # subplot titles
    if these_keys == ["Title"]:
        subplot_titles = None  # edge case, we entered a df with no title
    else:
        subplot_titles = these_keys

    # create plot
    if len(these_keys) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(these_keys)
    graph_num_rows = math.ceil((len(these_keys)) / graph_num_cols)

    fig = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    graph_r = 0
    graph_c = 0

    # for each building / subplot
    for this_key in these_keys:
        # progress subplot
        if graph_c == graph_num_cols:
            graph_c = 0
            graph_r += 1
        # grab data
        this_df = y_data[this_key]
        if y_error_up_data is not None:
            this_y_error_up = y_error_up_data[this_key]
        if y_error_down_data is not None:
            this_y_error_down = y_error_down_data[this_key]

        # sort
        if sort_by == "index":
            this_df = this_df.reset_index()  # Reset index to make it a column
            this_df["numeric_index"] = (
                this_df["index"].str.extract(r"(\d+)$").astype(int)
            )  # Extract the numeric part

            # Sort by the numeric part and then drop the helper column
            this_df = this_df.sort_values(by="numeric_index", ascending=ascending).drop(
                columns="numeric_index"
            )

            # Set the original index column back as the index
            this_df.set_index("index", inplace=True)
            this_df.index.name = None
        elif sort_by not in ["all", "dont"]:
            if sort_by is None:
                sort_by = list(this_df.columns)[0]
            this_df.sort_values(sort_by, inplace=True, ascending=ascending)

        # graph this subplot
        for this_col_i in range(num_y_cols):
            # grab data
            ser = this_df[y_cols[this_col_i]]
            if sort_by == "all":
                ser = ser.sort_values(ascending=ascending)
            ser_up = None
            if y_error_up_data is not None:
                ser_up = this_y_error_up.iloc[:, this_col_i]
            ser_down = None
            if y_error_down_data is not None:
                ser_down = this_y_error_down.iloc[:, this_col_i]
            for zone_i in range(len(ser.index)):
                zone = ser.index[zone_i]
                color, o, shape = "black", 1, "circle"  # defaults

                # color
                if (num_y_cols > 1) and (not color_override):
                    # mode 1: color by column (overrides color_data)
                    if (color_legend is not None) and ("color" in color_legend):
                        color = color_legend["color"][y_cols[this_col_i]]
                    else:
                        color = COLORS[this_col_i]
                elif color_data is not None:
                    # mode 2: we entered in color data
                    if len(list(color_data[these_keys[0]].columns)) == 1:
                        # one color mapping
                        this_color_col_i = 0
                    else:
                        this_color_col_i = this_col_i
                    try:
                        color_num = color_data[this_key].loc[
                            zone, color_data[this_key].columns[this_color_col_i]
                        ]
                    except Exception:
                        color_num = np.nan
                    if np.isnan(color_num):
                        color = "black"
                    else:
                        if (color_legend is not None) and ("color" in color_legend):
                            color = color_legend["color"][color_num]
                        else:
                            color = COLORS[color_num]
                color_error_bars = mcolors.to_rgba(color, alpha=0.5)
                color_error_bars = f"rgba({int(color_error_bars[0]*255)}, {int(color_error_bars[1]*255)}, {int(color_error_bars[2]*255)}, {color_error_bars[3]})"

                # opacity
                if opacity_data is not None:
                    if len(list(opacity_data[these_keys[0]].columns)) == 1:
                        # one opacity mapping
                        this_opacity_col_i = 0
                    else:
                        this_opacity_col_i = this_col_i
                    try:
                        o = opacity_data[this_key].loc[
                            zone, opacity_data[this_key].columns[this_opacity_col_i]
                        ]
                    except Exception:
                        o = 1
                    if np.isnan(o):
                        o = 0.01

                #  shape
                if shape_data is not None:
                    if len(list(shape_data[these_keys[0]].columns)) == 1:
                        # one shape mapping
                        this_shape_col_i = 0
                    else:
                        this_shape_col_i = this_col_i
                    try:
                        shape_num = shape_data[this_key].loc[
                            zone, shape_data[this_key].columns[this_shape_col_i]
                        ]
                    except Exception:
                        shape_num = np.nan
                    if np.isnan(shape_num):
                        shape = "asterisk"
                    else:
                        if (shape_legend is not None) and ("shape" in shape_legend):
                            shape = shape_legend["shape"][shape_num]
                        else:
                            shape = SHAPES[shape_num]

                # size
                if size_data is not None:
                    marker_size = size_data[this_key].loc[zone, :].iloc[0]

                # grab zonal values
                y_val = [ser[zone]]
                up_val, down_val = get_err_values(ser_up, ser_down, zone)

                # labels
                if (
                    (labels is not None)
                    and (this_key in labels)
                    and (zone in labels[this_key])
                ):
                    label = [zone]
                else:
                    label = None

                if normalize_x:
                    x = zone_i / len(ser.index)
                else:
                    x = zone_i

                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=y_val,
                        error_y=dict(
                            type="data",
                            array=up_val,
                            arrayminus=down_val,
                            color=color_error_bars,
                        ),
                        mode="markers+text",
                        marker_color=color,
                        opacity=o,
                        marker_size=marker_size,
                        marker_symbol=shape,
                        textposition="top center",
                        textfont=dict(size=14),
                        name=zone,
                        text=label,
                        showlegend=False,
                    ),
                    row=graph_r + 1,
                    col=graph_c + 1,
                )
        graph_c += 1

    # legend
    showlegend = False

    # colors
    if color_data is not None:
        _, max_color = find_min_max(color_data)
    if (num_y_cols > 1) and (not color_override):
        these_color_keys = y_cols
    elif color_data is not None:
        these_color_keys = list(range(int(max_color + 1)))
    else:
        these_color_keys = None
    if these_color_keys is not None:
        showlegend = True
        if color_legend_order is None:
            color_legend_order = list(range(len(these_color_keys)))
        for this_color_key_i in color_legend_order:
            this_color_key = these_color_keys[this_color_key_i]

            if this_color_key in dont_add_to_legend:
                continue
            if (color_legend is not None) and ("name" in color_legend):
                name = color_legend["name"][this_color_key]
            else:
                name = this_color_key
            if (color_legend is not None) and ("color" in color_legend):
                color = color_legend["color"][this_color_key]
            else:
                color = COLORS[this_color_key_i]

            if name is None:
                continue
            if name in dont_add_to_legend:
                continue

            shape = "circle"
            if color_and_shape:
                if (shape_legend is not None) and ("shape" in shape_legend):
                    shape = shape_legend["shape"][this_color_key]

            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    name=name,
                    marker_color=color,
                    marker_size=marker_size,
                    marker_symbol=shape,
                    showlegend=True,
                )
            )
    # shapes
    if (not color_and_shape) and (shape_data is not None):
        _, max_shape = find_min_max(shape_data)
        showlegend = True
        these_shape_keys = list(range(int(max_shape + 1)))
        if shape_legend_order is None:
            shape_legend_order = list(range(len(these_shape_keys)))
        for this_shape_key_i in shape_legend_order:
            this_shape_key = these_shape_keys[this_shape_key_i]
            if this_shape_key in dont_add_to_legend:
                continue
            if (shape_legend is not None) and ("name" in shape_legend):
                name = shape_legend["name"][this_shape_key]
            else:
                name = this_shape_key
            if (shape_legend is not None) and ("shape" in shape_legend):
                shape = shape_legend["shape"][this_shape_key]
            else:
                shape = SHAPES[this_shape_key_i]
            if name is None:
                continue
            if name in dont_add_to_legend:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    name=name,
                    marker_color="black",
                    marker_size=marker_size,
                    marker_symbol=shape,
                    showlegend=True,
                )
            )

    if y_range is None and force_same_yaxes:
        min_y_value, max_y_value = find_min_max(y_data)
        y_range = [
            adjust_axis_bound("min", min_y_value),
            adjust_axis_bound("max", max_y_value),
        ]

    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        showlegend=showlegend,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        grid_color=grid_color,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        x_rangemode="normal",
        y_rangemode="normal",
        y_range=y_range,
        x_range=x_range,
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
    )
    return fig


def make_scatter_plot(
    y_data,
    x_data,
    y_error_up_data=None,
    y_error_down_data=None,
    x_error_right_data=None,
    x_error_left_data=None,
    color_data=None,
    shape_data=None,
    opacity_data=None,
    size_data=None,
    color_and_shape=False,
    color_legend=None,
    shape_legend=None,
    color_legend_order=None,
    shape_legend_order=None,
    dont_add_to_legend=[],
    labels=None,
    title="Title",
    y_axis_title="Y Label",
    x_axis_title="X Label",
    force_same_yaxes=True,
    force_same_xaxes=True,
    y_range=None,
    x_range=None,
    y_zerolinecolor="Black",
    x_zerolinecolor="Black",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    marker_size=MRKR_SZ,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    General purpose scatter plot function

    Parameters
    ----------
    y_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        each df should have the same columns
        key is the subplot title
        columns are the variables plotted on each plot
    x_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        each df should have the same columns
        key is the subplot title
        dfs should have 1 column (same x shared by all ys)
        Or the same number of columns as y_data (each y col get own x col)
    y_error_up_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        optional to add error bars in up direction
        distance from central estimate
        must have same number of keys and columns as y_data
    y_error_down_data : dict(pd.DataFrame) or pd.DataFrame
        {key: pd.DataFrame}
        optional to add error bars in down direction
        distance from central estimate
        must have same number of keys and columns as y_data
    x_error_right_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        optional to add error bars in right direction
        distance from central estimate
        dfs must have same number of keys and columns as x_data
    x_error_left_data : dict(pd.DataFrame) or pd.DataFrame
        {key: pd.DataFrame}
        optional to add error bars in left direction
        distance from central estimate
        dfs must have same number of keys and columns as x_data
    color_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to color zones by data
        values should start at 0
        dfs should have 1 column (typical), or must have same number of columns as y_data
    shape_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to shape zones by data
        values should start at 0
        dfs should have 1 column, or must have same number of columns as y_data
    opacity_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to opacity zones by data
        dfs should have 1 column, or must have same number of columns as y_data
    size_data : dict or pd.DataFrame
        {key : pd.DataFrame}
        optional to change marker size by zone
        dfs should have 1 column
    color_and_shape : bool
        if True, use the same data for color and shape
        and combine in the legend
    color_legend : dict
        example:
        {"name" : {key1 : "Dominant", key2 : "Dominated"},
         "color" : {key1 : "blue", key2 : "red"}}
    shape_legend : dict
        example:
        {"name" : {key1 : "Included", key2 : "Excluded"},
         "shape" : {key1 : "circle", key2 : "x"}}
    color_legend_order : list
        list of ints determing order to add to legend
        default None, in ascending order
    shape_legend_order : list
        list of ints determing order to add to legend
        default None, in ascending order
    dont_add_to_legend : list
        list of vars not to add to legend
    labels : dict
        {key : list}
        for each key list of labels to show
    title : str
        only used if plotting one building
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default = "X Label"
    force_same_yaxes : bool
        whether all subplots have same y axis
    force_same_xaxes : bool
        whether all subplots have same x axis
        default = True
    y_range : list
        if used, we force y axis range to [min, max]
    x_range : list
       if used, we force x axis range to [min, max]
    y_zerolinecolor : str
        color or x axis (y zero line)
        default = "Black"
    x_zerolinecolor : str
        color of y axis (x zero line)
        default = "Black"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    marker_size : int
        size of markers
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Scatter plot figure

    Notes
    -----
    Works with df or dict(dfs)
    df gives you one plot, dict(df) gives you multiple subplots
    """
    if color_and_shape:
        shape_data = color_data

    # force into nested
    y_data = force_dict(y_data, title)
    x_data = force_dict(x_data, title)
    y_error_up_data = force_dict(y_error_up_data, title)
    y_error_down_data = force_dict(y_error_down_data, title)
    x_error_right_data = force_dict(x_error_right_data, title)
    x_error_left_data = force_dict(x_error_left_data, title)
    color_data = force_dict(color_data, title)
    shape_data = force_dict(shape_data, title)
    opacity_data = force_dict(opacity_data, title)
    size_data = force_dict(size_data, title)
    labels = force_dict(labels, title)

    # define cols
    these_keys = list(y_data.keys())
    y_cols = list(y_data[these_keys[0]].columns)
    num_y_cols = len(y_cols)

    # clean x and y
    for key in these_keys:
        y_data[key], x_data[key] = trim_to_common_elements(
            [y_data[key], x_data[key]], clean_idx=True, clean_cols=False
        )

    # subplot titles
    if these_keys == ["Title"]:
        subplot_titles = None  # edge case, we entered a df with no title
    else:
        subplot_titles = these_keys

    # create plot
    if len(these_keys) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(these_keys)
    graph_num_rows = math.ceil((len(these_keys)) / graph_num_cols)

    fig = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    graph_r = 0
    graph_c = 0
    # for each building / subplot
    for this_key in these_keys:
        # progress subplot
        if graph_c == graph_num_cols:
            graph_c = 0
            graph_r += 1
        # grab data
        this_df_y = y_data[this_key]
        if y_error_up_data is not None:
            this_y_error_up = y_error_up_data[this_key]
        if y_error_down_data is not None:
            this_y_error_down = y_error_down_data[this_key]

        this_df_x = x_data[this_key]
        if x_error_right_data is not None:
            this_x_error_right = x_error_right_data[this_key]
        if x_error_left_data is not None:
            this_x_error_left = x_error_left_data[this_key]

        # graph this subplot
        for this_col_i in range(num_y_cols):
            # grab data
            ser_y = this_df_y[y_cols[this_col_i]]
            ser_up = None
            if y_error_up_data is not None:
                ser_up = this_y_error_up.iloc[:, this_col_i]
            ser_down = None
            if y_error_down_data is not None:
                ser_down = this_y_error_down.iloc[:, this_col_i]

            if len(list(x_data[these_keys[0]].columns)) == 1:
                # can plot multiple ys on same x
                this_x_col_i = 0
            else:
                this_x_col_i = this_col_i
            ser_x = this_df_x.iloc[:, this_x_col_i]

            ser_right = None
            if x_error_right_data is not None:
                ser_right = this_x_error_right.iloc[:, this_x_col_i]
            ser_left = None
            if x_error_left_data is not None:
                ser_left = this_x_error_left.iloc[:, this_x_col_i]

            for zone_i in range(len(ser_y.index)):
                zone = ser_y.index[zone_i]
                color, o, shape = "black", 1, "circle"  # defaults
                # color
                if num_y_cols > 1:
                    # mode 1: color by column (overrides color_data)
                    if (color_legend is not None) and ("color" in color_legend):
                        color = color_legend["color"][y_cols[this_col_i]]
                    else:
                        color = COLORS[this_col_i]
                elif color_data is not None:
                    # mode 2: we entered in color data
                    if len(list(color_data[these_keys[0]].columns)) == 1:
                        # one color mapping
                        this_color_col_i = 0
                    else:
                        this_color_col_i = this_col_i
                    try:
                        color_num = color_data[this_key].loc[
                            zone, color_data[this_key].columns[this_color_col_i]
                        ]
                    except Exception:
                        color_num = np.nan
                    if np.isnan(color_num):
                        color = "black"
                    else:
                        if (color_legend is not None) and ("color" in color_legend):
                            color = color_legend["color"][color_num]
                        else:
                            color = COLORS[color_num]

                color_error_bars = mcolors.to_rgba(color, alpha=0.5)
                color_error_bars = f"rgba({int(color_error_bars[0]*255)}, {int(color_error_bars[1]*255)}, {int(color_error_bars[2]*255)}, {color_error_bars[3]})"

                # opacity
                if opacity_data is not None:
                    if len(list(opacity_data[these_keys[0]].columns)) == 1:
                        # one opacity mapping
                        this_opacity_col_i = 0
                    else:
                        this_opacity_col_i = this_col_i
                    try:
                        o = opacity_data[this_key].loc[
                            zone, opacity_data[this_key].columns[this_opacity_col_i]
                        ]
                    except Exception:
                        o = 1
                    if np.isnan(o):
                        o = 0.01

                #  shape
                if shape_data is not None:
                    if len(list(shape_data[these_keys[0]].columns)) == 1:
                        # one shape mapping
                        this_shape_col_i = 0
                    else:
                        this_shape_col_i = this_col_i
                    try:
                        shape_num = shape_data[this_key].loc[
                            zone, shape_data[this_key].columns[this_shape_col_i]
                        ]
                    except Exception:
                        shape_num = np.nan
                    if np.isnan(shape_num):
                        shape = "asterisk"
                    else:
                        if (shape_legend is not None) and ("shape" in shape_legend):
                            shape = shape_legend["shape"][shape_num]
                        else:
                            shape = SHAPES[shape_num]

                # size
                if size_data is not None:
                    marker_size = size_data[this_key].loc[zone, :].iloc[0]

                # grab zonal values
                y_val = [ser_y[zone]]
                x_val = [ser_x[zone]]
                up_val, down_val = get_err_values(ser_up, ser_down, zone)
                right_val, left_val = get_err_values(ser_right, ser_left, zone)

                # labels
                if (
                    (labels is not None)
                    and (this_key in labels)
                    and (zone in labels[this_key])
                ):
                    label = [zone]
                else:
                    label = None

                fig.add_trace(
                    go.Scatter(
                        x=x_val,
                        y=y_val,
                        error_x=dict(
                            type="data",
                            array=right_val,
                            arrayminus=left_val,
                            color=color_error_bars,
                        ),
                        error_y=dict(
                            type="data",
                            array=up_val,
                            arrayminus=down_val,
                            color=color_error_bars,
                        ),
                        mode="markers+text",
                        opacity=o,
                        marker_size=marker_size,
                        marker_symbol=shape,
                        name=zone,
                        text=label,
                        textposition="bottom left",
                        textfont=dict(size=14),
                        marker_color=color,
                        showlegend=False,
                    ),
                    row=graph_r + 1,
                    col=graph_c + 1,
                )
        graph_c += 1

    # legend
    showlegend = False

    # colors
    if color_data is not None:
        _, max_color = find_min_max(color_data)
    if num_y_cols > 1:
        these_color_keys = y_cols
    elif color_data is not None:
        these_color_keys = list(range(int(max_color + 1)))
    else:
        these_color_keys = None
    if these_color_keys is not None:
        showlegend = True
        if color_legend_order is None:
            color_legend_order = list(range(len(these_color_keys)))
        for this_color_key_i in color_legend_order:
            this_color_key = these_color_keys[this_color_key_i]

            if this_color_key in dont_add_to_legend:
                continue
            if (color_legend is not None) and ("name" in color_legend):
                name = color_legend["name"][this_color_key]
            else:
                name = this_color_key
            if (color_legend is not None) and ("color" in color_legend):
                color = color_legend["color"][this_color_key]
            else:
                color = COLORS[this_color_key_i]

            if name is None:
                continue
            if name is dont_add_to_legend:
                continue

            shape = "circle"
            if color_and_shape:
                if (shape_legend is not None) and ("shape" in shape_legend):
                    shape = shape_legend["shape"][this_color_key]

            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    name=name,
                    marker_color=color,
                    marker_size=marker_size,
                    marker_symbol=shape,
                    showlegend=True,
                )
            )
    # shapes
    if (not color_and_shape) and (shape_data is not None):
        _, max_shape = find_min_max(shape_data)
        showlegend = True
        these_shape_keys = list(range(int(max_shape + 1)))
        if shape_legend_order is None:
            shape_legend_order = list(range(len(these_shape_keys)))
        for this_shape_key_i in shape_legend_order:
            this_shape_key = these_shape_keys[this_shape_key_i]
            if this_shape_key in dont_add_to_legend:
                continue
            if (shape_legend is not None) and ("name" in shape_legend):
                name = shape_legend["name"][this_shape_key]
            else:
                name = this_shape_key
            if (shape_legend is not None) and ("shape" in shape_legend):
                shape = shape_legend["shape"][this_shape_key]
            else:
                shape = SHAPES[this_shape_key_i]
            if name is None:
                continue
            if name is dont_add_to_legend:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    name=name,
                    marker_color="black",
                    marker_size=marker_size,
                    marker_symbol=shape,
                    showlegend=True,
                )
            )

    if y_range is None and force_same_yaxes:
        min_y_value, max_y_value = find_min_max(y_data)
        y_range = [
            adjust_axis_bound("min", min_y_value),
            adjust_axis_bound("max", max_y_value),
        ]

    if x_range is None and force_same_xaxes:
        min_x_value, max_x_value = find_min_max(x_data)
        x_range = [
            adjust_axis_bound("min", min_x_value),
            adjust_axis_bound("max", max_x_value),
        ]

    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        showlegend=showlegend,
        y_axis_title=y_axis_title,
        x_axis_title=x_axis_title,
        grid_color=grid_color,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        y_range=y_range,
        x_range=x_range,
        y_rangemode="normal",
        x_rangemode="normal",
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
    )
    return fig


def make_time_series(
    y_data,
    y_error_up_data=None,
    y_error_down_data=None,
    start_date=None,
    end_date=None,
    equips=None,
    sort_equips=True,
    stack=False,
    x_axis_type="date",
    secondary_variables=None,
    line_legend=None,
    dont_add_to_legend=[],
    y_axis_title="Y Label",
    x_axis_title="",
    secondary_y_axis_title="Y Label",
    force_same_yaxes=True,
    y_range=None,
    secondary_y_range=None,
    y_zerolinecolor="LightGray",
    x_zerolinecolor="LightGray",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=None,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    line_width=3,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    Makes a time series plot for specified zones/ahus

    Parameters
    ----------
    y_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        key is the variable, each makes a line in each subplot
        df columns are equips, one subplot for each equip
        e.g. {"Temperature", df1, "CSP", df2}
    y_error_up_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        optional to add error haze in up direction
        distance from central estimate
        must have same number of keys and columns as y_data
    y_error_down_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        optional to add error haze in down direction
        distance from central estimate
        must have same number of keys and columns as y_data
    start_date : pd.Timestamp()
        where to start time series plot
        default = None, start at beginning of dataset
    end_date : pd.Timestamp()
        where to end time series plot
        default = None, finish at end of dataset
    equips : list of strings
        optional param to look at specific zones/equips
        default = None, all equips
    sort_equips : bool
        if True, sort the equips/cols alphabetically
    stack : bool
        whether to stack the lines
    x_axis_type : str
        'date' or 'category'
        use 'category' if you want to treat index as string
    secondary_variables : list(str)
        variable to plot on secondary y axis
        default = None
    line_legend : dict
        example:
        {"name" : {key1 : "DAT (F)", key2 : "Cooling Requests"},
         "color" : {key1 : "blue", key2 : "red"}
         "style" : {key1 : "solid", key2 : "dash"}
         "opacity" : {key1 : 1, key2 : 0.5}};
        style options 'solid', 'dash', 'dot', and 'dashdot'
    dont_add_to_legend : list
        list of variables to not add to legend
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default = ""
    secondary_y_axis_title : str
        y label for secondary y axis
        default = "Y Label"
    force_same_yaxes : bool
        whether all subplots have same y axis
        default = True
    y_range : list
        if used, we force y axis range to [min, max]
    secondary_y_range : list
        if used, we force secondary y axis range to [min, max]
    y_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    x_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    line_width : float
        controls line width
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Time series plot

    Notes
    -----
    Works with df or dict(dfs)
    df gives you one variable, dict(df) gives you multiple variables
    Warning: shading makes plot heavy and slow with all zones on
    """
    # force dict if pd.DataFrame
    if not isinstance(y_data, dict):
        y_data = {y_axis_title: y_data}

    # clean up and down
    if (y_error_up_data is not None) and (y_error_down_data is None):
        y_error_down_data = y_error_up_data
    if (y_error_down_data is not None) and (y_error_up_data is None):
        y_error_up_data = y_error_down_data

    if y_error_up_data is not None:
        if not isinstance(y_error_up_data, dict):
            y_error_up_data = {y_axis_title: y_error_up_data}
    if y_error_down_data is not None:
        if not isinstance(y_error_down_data, dict):
            y_error_down_data = {y_axis_title: y_error_down_data}

    # trim start and end date
    these_vars = list(y_data.keys())
    if start_date is None:
        start_date = y_data[these_vars[0]].index[0]
    if end_date is None:
        end_date = y_data[these_vars[0]].index[-1]
    for this_var in these_vars:
        y_data[this_var] = y_data[this_var].loc[start_date:end_date, :]

    # trim cols
    common = functools.reduce(
        lambda left, right: left.intersection(right, sort=sort_equips),
        [y_data[i].columns for i in y_data],
    )
    if equips is not None:
        equips_old = copy.deepcopy(equips)
        equips = list(set(common).intersection(set(equips)))
        if len(equips) < len(equips_old):
            print("Following equips not compatible with vars:")
            print(list(set(equips_old) - set(equips)))
    else:
        equips = common

    # plot specs
    num_equips = len(equips)
    if num_equips >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = num_equips
    graph_num_rows = math.ceil(num_equips / graph_num_cols)
    specs = [
        [
            {"secondary_y": secondary_variables is not None}
            for _ in range(graph_num_cols)
        ]
        for _ in range(graph_num_rows)
    ]

    plt = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=equips,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        specs=specs,
    )

    stackgroup_name = None
    stack_fill = None
    if stack:
        stackgroup_name = "one"
        stack_fill = "tonexty"
        # stack_fill = "tozeroy"

    # plot each var
    for i in range(len(these_vars)):
        this_var = these_vars[i]
        graph_c = 0
        graph_r = 0

        # line styles
        if line_legend is not None and "name" in line_legend:
            n = line_legend["name"][this_var]
        else:
            n = this_var
        if line_legend is not None and "color" in line_legend:
            c = line_legend["color"][this_var]
        else:
            c = COLORS[i]
        if line_legend is not None and "style" in line_legend:
            s = line_legend["style"][this_var]
            if s == "solid":
                s = None
        else:
            s = None
        if line_legend is not None and "opacity" in line_legend:
            o = line_legend["opacity"][this_var]
        else:
            o = 1

        # plot each equip
        for equip in equips:
            if graph_c == graph_num_cols:
                graph_c = 0
                graph_r += 1

            # add optional haze
            if y_error_up_data is not None:
                upper = (y_error_up_data[this_var][equip]).fillna(method="ffill")
                lower = (y_error_down_data[this_var][equip]).fillna(method="ffill")
                rgba = mcolors.to_rgba(c, alpha=0.3)
                rgba = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
                plt.add_trace(
                    go.Scatter(
                        x=list(y_data[this_var].index)
                        + list(y_data[this_var].index)[::-1],
                        y=list(upper) + list(lower)[::-1],
                        fill="toself",
                        fillcolor=rgba,
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                    ),
                    row=graph_r + 1,
                    col=graph_c + 1,
                    secondary_y=(
                        n in secondary_variables
                        if secondary_variables is not None
                        else False
                    ),
                )
            # add line
            plt.add_trace(
                go.Scatter(
                    x=y_data[this_var].index,
                    y=y_data[this_var][equip],
                    name=n,
                    mode="lines",
                    marker_color=c,
                    opacity=o,
                    showlegend=False,
                    line=dict(dash=s, width=line_width),
                    stackgroup=stackgroup_name,
                    fill=stack_fill,
                    fillcolor=c,
                ),
                row=graph_r + 1,
                col=graph_c + 1,
                secondary_y=(
                    n in secondary_variables
                    if secondary_variables is not None
                    else False
                ),
            )

            graph_c += 1
        # legend
        if (len(these_vars) > 1) and (this_var not in dont_add_to_legend):
            plt.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    marker_color=c,
                    opacity=o,
                    name=n,
                    line=dict(dash=s, width=line_width),
                    showlegend=True,
                )
            )

    # determine min / max
    if secondary_variables is not None:
        primary_variables = list(set(these_vars) - set(secondary_variables))
        min_y_value_1, max_y_value_1 = find_min_max(y_data, primary_variables)
        min_y_value_2, max_y_value_2 = find_min_max(y_data, secondary_variables)
    else:
        min_y_value_1, max_y_value_1 = find_min_max(y_data)

    if y_range is None and force_same_yaxes:
        y_range = [
            adjust_axis_bound("min", min_y_value_1),
            adjust_axis_bound("max", max_y_value_1),
        ]

    for ax in plt.layout:
        if ax.startswith("xaxis"):
            plt.layout[ax].type = x_axis_type

    plt = update_fig_formatting(
        plt,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        y_range=y_range,
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        grid_color=grid_color,
    )
    if secondary_variables is not None:
        plt.update_yaxes(
            title_text=secondary_y_axis_title,
            secondary_y=True,
        )
        plt.update_layout(
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", x=1.1)
        )
        if secondary_y_range is None and force_same_yaxes:
            secondary_y_range = [
                adjust_axis_bound("min", min_y_value_2),
                adjust_axis_bound("max", max_y_value_2),
            ]
        plt.update_yaxes(range=secondary_y_range, secondary_y=True)

    return plt


def plot_experiment_delta(
    experiment_results,
    experiment_covs,
    df,
    binary,
    dot_legend,
    mode="Absolute Change",
    summary_statistic="Mean",
    shape_legend=None,
    dont_add_to_legend=[],
    y_axis_title="Y Label",
    y_range=None,
    x_range=None,
    y_zerolinecolor="LightGray",
    x_zerolinecolor="LightGray",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    marker_size=MRKR_SZ,
    line_width=2,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    Plot experiment results

    Parameters
    ----------
    experiment_results : pd.DataFrame
        df with equips as index, cols as results
        output from general_Delta_fn
    experiment_covs : dict
        dict of dfs of covariances from general_Delta_fn
        keys are equips
        columns are from tests
    df : pd.DataFrame
        df with time as index (hourly) and equips as cols
        same input into general_Delta_fn
    binary : pd.Series or pd.DataFrame
        0 if not test day, 1 if test day
        each column represents a different test, e.g. CSP = 76F and CSP = 78F
    dot_legend : dict
        controls name and styles of dots/tests
        example
        dot_legend={
            "name": {"Control" : "Low SP", "Test 1" : "High SP"},
            "color": {"Control" : "Blue", "Test 1" : "Red"},
            "opacity" : {"Control" : 1, "Test 1" : 1}
        }
    mode : str
        "Absolute Change" or "Percent Change"
        default "Absolute Change"
    summary_statistic : str
        "Mean", "Sum", "Max", "Min", default "Mean"
    shape_legend : dict
        controls shape of dots
        example
        shape_legend={
            "series" : pd.Series(), idx is day and value is int encoding
            "name" : {0 : "Weekday", 1 : "Weekend", 2 : "Today"},
            "shape" : {0 : "circle", 1 : "x", 2 : "star"},
        }
    dont_add_to_legend : list
        columns or conditions etc that should not be added to legend
    y_axis_title : str
        y label
        default = "Y Label"
    y_range : list
        if used, we force y axis range to [min, max]
    x_range : list
        if used, we force x axis range to [min, max]
    y_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    x_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    marker_size : int
        marker size
    line_width : float
        line thickness
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Experiment figure
    """
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
    common = binary.index.intersection(df.index, sort=True)
    binary = binary.loc[common, :]
    df = df.loc[common, :]

    # initializations
    equips = list(df.columns)

    # prep plot
    if len(equips) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(equips)
    graph_num_rows = math.ceil(len(equips) / graph_num_cols)
    fig = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=equips,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    # conditions to plot
    idxs = {}
    conditions = list(binary.columns)
    control = binary[conditions].sum(axis=1)
    idxs["Control"] = list((control[control == 0]).index)
    for condition in conditions:
        bin_col = binary[condition]
        idxs[condition] = list((bin_col[bin_col == 1]).index)

    # plot for each condition
    c = 0
    for condition in idxs:
        if len(idxs[condition]) == 0:
            continue

        # reset plot counters
        graph_r, graph_c = 0, 0

        # plot each equip
        for equip in equips:
            # filter data for this equip
            ys = df[equip]
            ys = ys.dropna()

            if graph_c == graph_num_cols:
                graph_c = 0
                graph_r += 1

            # intersect condition with y
            these_idx = list(set(idxs[condition]).intersection(set(list(ys.index))))
            this_y = ys[these_idx]

            # first plot the dots
            color = COLORS[c]
            if "color" in dot_legend and condition in dot_legend["color"]:
                color = dot_legend["color"][condition]

            opacity = 0.5
            if "opacity" in dot_legend and condition in dot_legend["opacity"]:
                opacity = dot_legend["opacity"][condition]

            name = condition
            if "name" in dot_legend and condition in dot_legend["name"]:
                name = dot_legend["name"][condition]

            for idx in these_idx:
                if shape_legend is not None:
                    this_shape = shape_legend["shape"][shape_legend["series"][idx]]
                else:
                    this_shape = "circle"
                fig.add_trace(
                    go.Scatter(
                        x=[name],
                        y=[this_y[idx]],
                        name=str(idx),  # day
                        mode="markers",
                        marker_color=color,
                        marker_symbol=this_shape,
                        marker_size=marker_size,
                        opacity=opacity,
                        showlegend=False,
                    ),
                    row=graph_r + 1,
                    col=graph_c + 1,
                )

            # next plot the estimate with 95% confidence interval
            if c == 0:
                # control
                y_val = experiment_results.loc[equip, "Slope Intercept"]
                std_err = experiment_results.loc[equip, "Std Err Intercept"]
            else:
                y_val = (
                    experiment_results.loc[equip, "Slope Intercept"]
                    + experiment_results.loc[equip, f"Slope {condition}"]
                )
                try:
                    std_err = np.sqrt(
                        experiment_results.loc[equip, "Std Err Intercept"] ** 2
                        + experiment_results.loc[equip, f"Std Err {condition}"] ** 2
                        + 2 * experiment_covs[equip].loc["Intercept", condition]
                    )
                except Exception:
                    std_err = np.sqrt(
                        experiment_results.loc[equip, "Std Err Intercept"] ** 2
                        + experiment_results.loc[equip, f"Std Err {condition}"] ** 2
                        + 0
                    )

            y_up = y_val + 1.96 * std_err
            y_down = y_val - 1.96 * std_err
            if mode == "Percent Change":
                y_val = np.exp(y_val)
                y_up = np.exp(y_up)
                y_down = np.exp(y_down)

            fig.add_trace(
                go.Scatter(
                    x=[name],
                    y=[y_val],
                    error_y=dict(
                        type="data",
                        array=[y_up - y_val],
                        arrayminus=[y_val - y_down],
                        color=color,
                    ),
                    mode="markers",
                    marker_color=color,
                    opacity=1,
                    marker_size=marker_size + 4,
                    name=name,
                    showlegend=False,
                ),
                row=graph_r + 1,
                col=graph_c + 1,
            )

            graph_c += 1

        # color legend
        if condition not in dont_add_to_legend:
            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    name=name,
                    marker_color=color,
                    marker_size=marker_size,
                    showlegend=True,
                )
            )
        c += 1
    # shape legend
    if (shape_legend is not None) and ("name" in shape_legend):
        for shape_key in shape_legend["name"]:
            if shape_legend["name"][shape_key] not in dont_add_to_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[np.nan],
                        y=[np.nan],
                        mode="markers",
                        marker_size=marker_size,
                        name=shape_legend["name"][shape_key],
                        marker_color="black",
                        marker_symbol=shape_legend["shape"][shape_key],
                        showlegend=True,
                    )
                )
    # format
    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        y_axis_title=y_axis_title,
        x_axis_title="",
        y_rangemode="normal",
        x_rangemode="normal",
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
        y_range=y_range,
        x_range=x_range,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        grid_color=grid_color,
    )
    return fig


def plot_experiment_regression(
    experiment_results,
    df,
    T,
    binary,
    line_legend,
    mode="Percent Change",
    summary_statistic="Mean",
    shape_legend=None,
    additive_column_dict=None,
    dont_plot_lines=[],
    dont_plot_dots=[],
    dont_add_to_legend=[],
    y_axis_title="Y Label",
    x_axis_title="Average Daytime OAT (F)",
    y_range=None,
    x_range=None,
    y_zerolinecolor="LightGray",
    x_zerolinecolor="LightGray",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    marker_size=MRKR_SZ,
    line_width=2,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
    fig=None,
):
    """
    Plot experiment results

    Parameters
    ----------
    experiment_results : pd.DataFrame
        df with equips as index, cols as results
        output from general_Delta_fn
    df : pd.DataFrame
        df with time as index (hourly) and equips as cols
        same input into general_Delta_fn
    T : pd.Series
        outside temperature data of interest (hourly)
        same input into general_Delta_fn
    binary : pd.Series or pd.DataFrame
        0 if not test day, 1 if test day
        each column represents a different test, e.g. CSP = 76F and CSP = 78F
        same input into general_Delta_fn
    line_legend : dict
        controls name and styles of lines/tests
        example
        line_legend={
            "name": {"Control" : "Low SP", "Test 1" : "High SP"},
            "color": {"Control" : "Blue", "Test 1" : "Red"},
            "style": {"Control" : "solid", "Test 1" "solid"}
            "opacity" : {"Control" : 1, "Test 1" : 1}
        }
        stype options 'solid', 'dash', 'dot', 'dashdot'
    mode : str
        "Absolute Change" or "Percent Change"
        default "Percent Change"
    summary_statistic : str
        "Mean", "Sum", "Max", "Min", default "Mean"
    shape_legend : dict
        controls shape of dots
        example
        shape_legend={
            "series" : pd.Series(), idx is day and value is int encoding
            "name" : {0 : "Weekday", 1 : "Weekend", 2 : "Today"},
            "shape" : {0 : "circle", 1 : "x", 2 : "star"},
        }
    additive_column_dict : dict
        key of dict is the column to add, values are the columns to add it to
        column that when plotted is not its own feature, rather it is added to other columns
        regression is plotted with this column bin = 0 as solid line, bin = 1 dashed line
        limited to a single key (only one column to add)
    dont_plot_lines : list
        columns that should not be plotted with their own regression lines
    dont_plot_dots : list
        columns that should not be plotted with dots
    dont_add_to_legend : list
        columns or conditions etc that should not be added to legend
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        y label
        default = "Average Daytime OAT (F)"
    y_range : list
        if used, we force y axis range to [min, max]
    x_range : list
        if used, we force x axis range to [min, max]
    y_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    x_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    marker_size : int
        marker size
    line_width : float
        line thickness
    width : int
        width of sub-plots
    height : int
        height of sub-plots
    fig : optional fig
        can throw in existing plot so as to plot multiple regressions

    Returns
    -------
    Experiment figure
    """
    # allow for pd.Series input as binary
    if isinstance(binary, pd.Series):
        binary = binary.to_frame()
        binary.columns = ["High SP"]

    if isinstance(T, pd.DataFrame):
        T = T["temperature"]

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

    T = T.groupby(T.index.date).mean()
    T.index = pd.DatetimeIndex(T.index)
    binary.dropna(how="all", inplace=True)
    binary.index = pd.DatetimeIndex(binary.index)
    common = binary.index.intersection(df.index).intersection(T.index, sort=True)
    binary = binary.loc[common, :]
    df = df.loc[common, :]
    T = T.loc[common]
    T_range = np.arange(int(T.min()), int(T.max()), 0.1)
    # initializations
    equips = list(df.columns)

    # prep plot
    if len(equips) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(equips)
    graph_num_rows = math.ceil(len(equips) / graph_num_cols)
    if fig is None:
        fig = sbplt.make_subplots(
            rows=graph_num_rows,
            cols=graph_num_cols,
            subplot_titles=equips,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )

    # conditions to plot
    idxs = {}

    conditions = list(binary.columns)
    if additive_column_dict is not None:
        additive_column = list(additive_column_dict.keys())[0]
        conditions.remove(additive_column)

    control = binary[conditions].sum(axis=1)
    idxs["Control"] = list((control[control == 0]).index)
    for condition in conditions:
        bin_col = binary[condition]
        idxs[condition] = list((bin_col[bin_col == 1]).index)

    # plot for each condition
    for condition in idxs:
        if len(idxs[condition]) == 0:
            continue

        # reset plot counters
        graph_r, graph_c, c = 0, 0, 0

        # plot each equip
        for equip in equips:
            # filter data for this equip
            ys = df[equip]
            ys = ys.dropna()
            if len(ys) < 3:
                continue

            if graph_c == graph_num_cols:
                graph_c = 0
                graph_r += 1

            # intersect condition with y
            these_idx = list(set(idxs[condition]).intersection(set(list(ys.index))))
            this_y = ys[these_idx]
            this_x = T.loc[these_idx]
            color = COLORS[c]

            # first plot the dots
            if "color" in line_legend and condition in line_legend["color"]:
                color = line_legend["color"][condition]
            opacity = 1
            if "opacity" in line_legend and condition in line_legend["opacity"]:
                opacity = line_legend["opacity"][condition]
            if condition not in dont_plot_dots:
                for idx in these_idx:
                    if shape_legend is not None:
                        this_shape = shape_legend["shape"][shape_legend["series"][idx]]
                    else:
                        this_shape = "circle"
                    fig.add_trace(
                        go.Scatter(
                            x=[this_x[idx]],
                            y=[this_y[idx]],
                            name=str(idx),  # day
                            mode="markers",
                            marker_color=color,
                            marker_symbol=this_shape,
                            marker_size=marker_size,
                            opacity=opacity,
                            showlegend=False,
                        ),
                        row=graph_r + 1,
                        col=graph_c + 1,
                    )

            # next plot the regression lines
            if condition not in dont_plot_lines:
                temp_slope = experiment_results.loc[equip, "Slope OAT"]
                test_slope = 0
                if condition != "Control":
                    test_slope = experiment_results.loc[equip, f"Slope {condition}"]
                y_reg = (
                    T_range * temp_slope
                    + test_slope
                    + experiment_results.loc[equip, "Slope Intercept"]
                )
                if mode == "Percent Change":
                    y_reg = np.exp(y_reg)
                # line formatting
                name = condition
                if "name" in line_legend and condition in line_legend["name"]:
                    name = line_legend["name"][condition]
                line_style = None
                if "style" in line_legend and condition in line_legend["style"]:
                    line_style = line_legend["style"][condition]
                    if line_style == "solid":
                        line_style = None
                fig.add_trace(
                    go.Scatter(
                        x=T_range,
                        y=y_reg,
                        name=name,
                        mode="lines",
                        marker_color=color,
                        line=dict(dash=line_style, width=line_width),
                        opacity=opacity,
                        showlegend=False,
                    ),
                    row=graph_r + 1,
                    col=graph_c + 1,
                )
                # additive column line
                if (additive_column_dict is not None) and (
                    condition in additive_column_dict[additive_column]
                ):
                    if mode == "Percent Change":
                        y_reg *= np.exp(
                            experiment_results.loc[equip, f"Slope {additive_column}"]
                        )
                    else:
                        y_reg += experiment_results.loc[
                            equip, f"Slope {additive_column}"
                        ]
                    fig.add_trace(
                        go.Scatter(
                            x=T_range,
                            y=y_reg,
                            name=name,
                            mode="lines",
                            marker_color=color,
                            line=dict(dash=line_style, width=line_width),
                            opacity=opacity,
                            showlegend=False,
                        ),
                        row=graph_r + 1,
                        col=graph_c + 1,
                    )
            graph_c += 1
            c += 1
        # color legend
        if condition not in dont_add_to_legend:
            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="lines+markers",
                    name=name,
                    marker_color=color,
                    marker_size=marker_size,
                    line=dict(dash=line_style, width=line_width),
                    opacity=opacity,
                    showlegend=True,
                )
            )
    # shape legend
    if (shape_legend is not None) and ("name" in shape_legend):
        for shape_key in shape_legend["name"]:
            if shape_legend["name"][shape_key] not in dont_add_to_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[np.nan],
                        y=[np.nan],
                        mode="markers",
                        marker_size=marker_size,
                        name=shape_legend["name"][shape_key],
                        marker_color="black",
                        marker_symbol=shape_legend["shape"][shape_key],
                        showlegend=True,
                    )
                )
    # format
    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        y_axis_title=y_axis_title,
        x_axis_title=x_axis_title,
        y_rangemode="normal",
        x_rangemode="normal",
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
        y_range=y_range,
        x_range=x_range,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        grid_color=grid_color,
    )
    return fig


def plot_regression(
    regression_results,
    y_data,
    x_data,
    line_legend,
    summary_statistic="Mean",
    y_axis_title="Y Label",
    x_axis_title="X Label",
    y_range=None,
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    marker_size=MRKR_SZ,
    line_width=2,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
    fig=None,
):
    """
    Plot regression results

    Parameters
    ----------
    regression_results : pd.DataFrame
        df with equips as index, cols as results (slope, int, etc)
    y_data : pd.DataFrame
        y_data with time as index (hourly) and equips as cols
    x_data : pd.Series
        x_data with time as index (hourly) and equips as cols
    line_legend : dict
        controls name and color of regression
    summary_statistic : str
        "Mean", "Sum", or "Max", default "Mean"
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default = "X Label"
    y_range : list
        if used, we force y axis range to [min, max]
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    marker_size : int
        marker size
    line_width : float
        line thickness
    width : int
        width of sub-plots
    height : int
        height of sub-plots
    fig : optional fig
        can throw in existing plot so as to plot multiple experiments

    Returns
    -------
    Regression figure
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
    equips = list(y_data.columns)

    # prep plot
    if len(equips) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(equips)
    graph_num_rows = math.ceil(len(equips) / graph_num_cols)
    if fig is None:
        fig = sbplt.make_subplots(
            rows=graph_num_rows,
            cols=graph_num_cols,
            subplot_titles=equips,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    graph_r = 0
    graph_c = 0

    # plot each equip
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
            continue
        xs = xs.to_frame()
        xs["Intercept"] = pd.Series(1, index=xs.index)
        vars = ["X", "Intercept"]
        xs.columns = vars
        # plot
        if graph_c == graph_num_cols:
            graph_c = 0
            graph_r += 1
        # dots
        for idx in ys.index:
            fig.add_trace(
                go.Scatter(
                    x=[xs["X"][idx]],
                    y=[ys[idx]],
                    name=str(idx),
                    mode="markers",
                    marker_color=line_legend["color"],
                    showlegend=False,
                    marker_size=marker_size,
                ),
                row=graph_r + 1,
                col=graph_c + 1,
            )
        # regression
        x_sorted = xs["X"].sort_values()
        y_reg = (
            x_sorted * regression_results.loc[equip, "Slope X"]
            + regression_results.loc[equip, "Slope Intercept"]
        )
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_reg,
                name=line_legend["name"],
                mode="lines",
                marker_color=line_legend["color"],
                showlegend=False,
                line=dict(width=line_width),
            ),
            row=graph_r + 1,
            col=graph_c + 1,
        )
        graph_c += 1
    # legend
    fig.add_trace(
        go.Scatter(
            x=[np.nan],
            y=[np.nan],
            mode="markers",
            name=line_legend["name"],
            marker_color=line_legend["color"],
            showlegend=True,
            marker_size=marker_size,
        )
    )
    # format
    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        y_axis_title=y_axis_title,
        x_axis_title=x_axis_title,
        y_rangemode="normal",
        y_range=y_range,
    )
    return fig


def plot_experiment_summary(
    y_data,
    y_error_up_data=None,
    y_error_down_data=None,
    point_start=0,
    offset_delta=0.2,
    point_vals=None,
    tick_vals=None,
    secondary_variables=None,
    marker_legend=None,
    shape_legend=None,
    dont_add_to_legend=[],
    title="Title",
    y_axis_title="Y Label",
    secondary_y_axis_title="Y Label",
    force_same_yaxes=True,
    y_range=None,
    secondary_y_range=None,
    x_range=None,
    y_zerolinecolor="Black",
    x_zerolinecolor="LightGray",
    grid_color="LightGray",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    marker_size=MRKR_SZ,
    error_thickness=2,
    whisker_len=MRKR_SZ,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    Plot summary figure of experiments

    Parameters
    ----------
    y_data : pd.DataFrame or dict(pd.DataFrame)
        {key : pd.DataFrame}
        central estimate
        key is the subplot title
        df with equips as index, columns as experiments
        each df should have the same columns
        columns are the variables plotted on each plot
    y_error_up_data : pd.DataFrame or dict(pd.DataFrame)
        {key : pd.DataFrame}
        distance from top of 95% confidence interval to central estimate
        df with equips as index, columns as experiments
        must have same number of keys and columns as y_data
    y_error_down_data : pd.DataFrame oe dict(pd.DataFrame)
        {key : pd.DataFrame}
        distance from bottom of 95% confidence interval to central estimate
        df with equips index, columns as experiments
        must have same number of keys and columns as y_data
    point_start : float
        where to start data points
    offset_delta : float
        delta between points
    point_vals : list
        a different way to specify location of points, directly with list
    tick_vals : list
        list of coordinates for the building titles
        default None, auto-calculated
    secondary_variables : list
        which cols to send to secondary axis
    marker_legend : dict
        controls name, color, opacity, and marker shape of experiments
        example
        marker_legend={
            "color": {"76 2021": "grey", "76": "orange", "78": "red"},
            "name": {"76 2021": "CSP = 76F (Prev)", "76": "CSP = 76F", "78": "CSP = 78F"},
            "opacity": {"76 2021": 1, "76": 1, "78": 1},
            "shape": {"76 2021": "circle", "76": "circle", "78": "circle"}
        }
    shape_legend : dict
        only helps add black shape to legend
        example : {"Include" : "circle, "Excluded": "x"}
    dont_add_to_legend : list
        list of variables to not add to legend
    title : str
        only used if plotting one building
    y_axis_title : str
        y axis title
    secondary_y_axis_title : str
        y axis title
    force_same_yaxes : bool
        whether all subplots have same yaxes
        default = True
    y_range : list
        if used, we force y axis range to [min, max]
    secondary_y_range : list
        if used, we force secondary y axis range to [min, max]
    x_range : list
        if used, we force x axis range to [min, max]
    y_zerolinecolor : str
        color of x axis (y zero line)
        default = "Black"
    x_zerolinecolor : str
        color of x axis (y zero line)
        default = "LightGray"
    grid_color : str
        color of background grid
        default = "LightGray"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        text size of axes
    legend_size : int
        text size of legend
    marker_size : int
        size of markers
    error_thickness : float
        thickness of error bars
    whisker_len : int
        horizontal len of error whiskers
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Summary figure
    """
    # force nested
    y_data = force_dict(y_data, title)
    y_error_up_data = force_dict(y_error_up_data, title)
    y_error_down_data = force_dict(y_error_down_data, title)

    these_keys = list(y_data.keys())
    these_equips = list(y_data[these_keys[0]].index)
    these_cols = list(y_data[these_keys[0]].columns)

    if len(these_keys) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(these_keys)
    graph_num_rows = math.ceil((len(these_keys)) / graph_num_cols)

    # subplot titles
    if these_keys == ["Title"]:
        subplot_titles = None  # edge case, we entered a df with no title
    else:
        subplot_titles = these_keys

    specs = [
        [
            {"secondary_y": secondary_variables is not None}
            for _ in range(graph_num_cols)
        ]
        for _ in range(graph_num_rows)
    ]

    fig = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        specs=specs,
    )

    graph_row = 0
    graph_col = 0

    for key in these_keys:
        if graph_col == graph_num_cols:
            graph_col = 0
            graph_row += 1

        if (offset_delta > 0) and (len(these_cols) > (1 / offset_delta)):
            offset_delta = 1 / len(these_cols)
        for i in range(len(these_cols)):
            col = these_cols[i]
            offset = i * offset_delta + point_start
            # color
            if (marker_legend is not None) and ("color" in marker_legend):
                color = marker_legend["color"][col]
            else:
                color = COLORS[i]
            # name
            if (marker_legend is not None) and ("name" in marker_legend):
                name = marker_legend["name"][col]
            else:
                name = col
            # opacity
            if (marker_legend is not None) and ("opacity" in marker_legend):
                o = marker_legend["opacity"][col]
            else:
                o = 1
            # shape
            if (marker_legend is not None) and ("shape" in marker_legend):
                shape = marker_legend["shape"][col]
            else:
                shape = "circle"

            for j in range(len(these_equips)):
                equip = these_equips[j]
                if point_vals is None:
                    x = [j + offset]
                else:
                    x = [j + point_vals[i]]

                if y_error_up_data is None:
                    dh = None
                else:
                    dh = [y_error_up_data[key].loc[equip, col]]

                if y_error_down_data is None:
                    dl = None
                else:
                    dl = [y_error_down_data[key].loc[equip, col]]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=[y_data[key].loc[equip, col]],
                        error_y=dict(
                            type="data",
                            array=dh,
                            arrayminus=dl,
                            width=whisker_len,
                            thickness=error_thickness,
                        ),
                        mode="markers",
                        marker_color=color,
                        opacity=o,
                        marker_size=marker_size,
                        marker_symbol=shape,
                        showlegend=False,
                    ),
                    row=graph_row + 1,
                    col=graph_col + 1,
                    secondary_y=(
                        col in secondary_variables
                        if secondary_variables is not None
                        else False
                    ),
                )
            if graph_row == 0 and graph_col == 0:
                # only add to legend once
                if name not in dont_add_to_legend:
                    fig.add_trace(
                        go.Scatter(
                            x=[np.nan],
                            y=[np.nan],
                            mode="markers",
                            marker_color=color,
                            opacity=o,
                            marker_size=marker_size,
                            marker_symbol=shape,
                            name=name,
                            showlegend=True,
                        )
                    )

        graph_col += 1

    if shape_legend is not None:
        for condition in shape_legend:
            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    marker_color="black",
                    marker_size=marker_size,
                    marker_symbol=shape_legend[condition],
                    name=condition,
                    showlegend=True,
                )
            )

    if tick_vals is None:
        tick_vals = list(range(len(these_equips)))
    fig = fig.update_xaxes(tickvals=tick_vals, ticktext=these_equips)

    # determine min / max
    if secondary_variables is not None:
        primary_variables = list(set(these_cols) - set(secondary_variables))
        min_y_value_1, max_y_value_1 = find_min_max(y_data, primary_variables)
        min_y_value_2, max_y_value_2 = find_min_max(y_data, secondary_variables)
    else:
        min_y_value_1, max_y_value_1 = find_min_max(y_data)

    if y_range is None and force_same_yaxes:
        y_range = [
            adjust_axis_bound("min", min_y_value_1),
            adjust_axis_bound("max", max_y_value_1),
        ]

    fig = update_fig_formatting(
        fig,
        y_axis_title=y_axis_title,
        x_axis_title="",
        y_range=y_range,
        x_range=x_range,
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        grid_color=grid_color,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
    )
    if secondary_variables is not None:
        fig.update_yaxes(
            title_text=secondary_y_axis_title,
            secondary_y=True,
        )
        fig.update_layout(
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", x=1.1)
        )
        if secondary_y_range is None and force_same_yaxes:
            secondary_y_range = [
                adjust_axis_bound("min", min_y_value_2),
                adjust_axis_bound("max", max_y_value_2),
            ]
        fig.update_yaxes(range=secondary_y_range, secondary_y=True)
    return fig


def find_bar_min_max(y_data, bar_mode):
    min_y_value = 0
    max_y_value = -(10**6)
    if bar_mode == "group":
        max_dfs = copy.deepcopy(y_data)
    else:
        max_dfs = {}
        for key in y_data:
            max_dfs[key] = y_data[key].sum(axis=1).to_frame()
    for key in y_data:
        if max_dfs[key].max().max() > max_y_value:
            max_y_value = max_dfs[key].max().max()
    return min_y_value, max_y_value


def make_bar_plot(
    y_data,
    y_error_up_data=None,
    y_error_down_data=None,
    secondary_bars=[],
    secondary_y_data=None,
    bar_width=0.75,
    bar_gap=0.2,
    bar_group_gap=0.1,
    bar_mode="stack",
    bar_mode_dict=None,
    tick_vals=None,
    bar_legend=None,
    pattern_legend=None,
    dont_add_to_legend=[],
    title="Title",
    y_axis_title="Y Label",
    x_axis_title=None,
    secondary_y_axis_title="Y Label",
    annotations=[],
    annotation_type="float",
    annotation_angle=0,
    annotation_thresh=0.1,
    annotation_color="white",
    force_same_yaxes=True,
    y_range=None,
    secondary_y_range=None,
    y_zerolinecolor="White",
    x_zerolinecolor="White",
    grid_color="White",
    num_cols=GRAPH_NUM_COLS,
    horizontal_spacing=HORIZONTAL_SPACING,
    vertical_spacing=VERTICAL_SPACING,
    title_size=TITLE_SZ,
    text_size=TXT_SZ,
    legend_size=LEGEND_SZ,
    legend_order="reversed",
    annotation_size=18,
    line_width=3,
    marker_size=MRKR_SZ,
    legend_marker_size=MRKR_SZ,
    error_thickness=2,
    whisker_len=MRKR_SZ,
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
):
    """
    General purpose bar plot function

    Parameters
    ----------
    y_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        each df should have the same idx and cols
        key is the subplot title
        idx is the x-axis, cols are the sub-categories
    y_error_up_data : pd.DataFrame or dict(pd.DataFrame)
        {key : pd.DataFrame}
        distance from top of 95% confidence interval to central estimate
        must have same keys, idx, and cols as y_data
        only available in group mode (not stack)
    y_error_down_data : pd.DataFrame oe dict(pd.DataFrame)
        {key : pd.DataFrame}
        distance from bottom of 95% confidence interval to central estimate
        must have same keys, idx, and cols as y_data
        only available in group mode (not stack)
    secondary_bars : list
        list of groups/rows to use a secondary axis
    secondary_y_data : dict(pd.DataFrame) or pd.DataFrame
        {key : pd.DataFrame}
        for line data overlayed on top of bars
        has same keys and x axis as y_data
        typically only has one column
    bar_width : float
        width of bars
    bar_gap : float
        spacing between groups of bars
    bar_group_gap : float
        spacing between bars within the same group
    bar_mode : str
        either "group" or "stack"
        default "stack"
    bar_mode_dict : dict
        {col_name: group_name}
    tick_vals : list
        list of coordinates for x axis titles
        default None, auto-calculated
    bar_legend : dict
        example
        bar_legend = {
            "name" : {col1 : "Dominant", col2 : "Dominated"},
            "color" : {col1 : "blue", col2 " "red"},
            "opacity" : {col1 : 1, col2 " 1},
            "pattern" : {col1 : '', col2: '/'}
        }
        different shape options: '', '/', '\\', 'x', '-', '|', '.', '+'
    pattern_legend : dict
        only helps add black pattern to legend
        example : {"Included" : ('', 1), "Excluded": ('x', 1)}
        the tuple is (pattern, opacity)
    dont_add_to_legend : list
        list not to add to legend
    title : str
        only used if plotting one building
    y_axis_title : str
        y label
        default = "Y Label"
    x_axis_title : str
        x label
        default None
    secondary_y_axis_title : str
        y label
        default = "Y Label"
    annotations : list
        add annotations to categories in the list
        we only add annotation if the sub-bar is large enough
    annotation_type : str
        "int" or "float"
    annotation_angle : int
        angle of annotation
        0 is horizontal, 90 vertical
    annotation_thresh : float
        if smaller than this thresh, dont annotate
    annotation_color : str
        "black" or "white"
    force_same_yaxes : bool
        whether all subplots have same y axis
    y_range : list
         if used, we force y axis range to [min, max]
    secondary_y_range : list
         if used, we force secondary y axis range to [min, max]
    y_zerolinecolor : str
        color of x axis (y zero line)
        default = "White"
    x_zerolinecolor : str
        color of y axis (x zero line)
        default = "White"
    grid_color : str
        color of background grid
        default = "White"
    num_cols : int
        number of columns in plot
    horizontal_spacing : float
        horizontal spacing between subplots
    vertical_spacing : float
        vertical spacing between subplots
    title_size : int
        size of title
    text_size : int
        size of text in axes
    legend_size : int
        size of text in legend
    legend_order : str
        "reversed", "forward"
    annotation_size : int
        size of annotation text
    line_width : int
        width of line when plotting secondary line
    marker_size : int
        size of marker when plotting secondary line
    legend_marker_size : int
        size of box in legend
    error_thickness : float
        thickness of error bars
    whisker_len : int
        horizontal len of error whiskers
    width : int
        width of sub-plots
    height : int
        height of sub-plots

    Returns
    -------
    Bar plot figure

    Notes
    -----
    Assumes y_data have same idx and cols
    """
    # force nested
    y_data = force_dict(y_data, title)

    if y_error_up_data is not None:
        y_error_up_data = force_dict(y_error_up_data, title)

    if y_error_down_data is not None:
        y_error_down_data = force_dict(y_error_down_data, title)

    if secondary_y_data is not None:
        secondary_y_data = force_dict(secondary_y_data, title)

    # define specs
    these_keys = list(y_data.keys())
    groups = list(y_data[these_keys[0]].columns)
    num_groups = len(groups)
    idxs = list(y_data[these_keys[0]].index)
    idxs_i = list(range(len(idxs)))

    if tick_vals is None:
        tick_vals = idxs_i

    if bar_mode_dict is None:
        if bar_mode == "stack":
            bar_mode_dict = {group: "__all__" for group in groups}
        elif bar_mode == "group":
            bar_mode_dict = {group: group for group in groups}

    stack_names = []
    for group in groups:
        stack_name = bar_mode_dict[group]
        if stack_name not in stack_names:
            stack_names.append(stack_name)

    num_stacks = len(stack_names)
    stack_positions = {stack_name: i for i, stack_name in enumerate(stack_names)}
    dx = bar_width / num_stacks

    # tidy color legend
    if bar_legend is None:
        bar_legend = {"name": {}, "color": {}, "opacity": {}, "pattern": {}}
        for g in range(len(groups)):
            bar_legend["name"][groups[g]] = groups[g]
            bar_legend["color"][groups[g]] = COLORS[g]
            bar_legend["opacity"][groups[g]] = 1
            bar_legend["pattern"][groups[g]] = ""
    else:
        if "name" not in bar_legend:
            bar_legend["name"] = {}
            for group in groups:
                bar_legend["name"][group] = group
        if "color" not in bar_legend:
            bar_legend["color"] = {}
            for g in range(len(groups)):
                bar_legend["color"][groups[g]] = COLORS[g]
        if "opacity" not in bar_legend:
            bar_legend["opacity"] = {}
            for g in range(len(groups)):
                bar_legend["opacity"][groups[g]] = 1
        if "pattern" not in bar_legend:
            bar_legend["pattern"] = {}
            for g in range(len(groups)):
                bar_legend["pattern"][groups[g]] = ""

    # subplot titles
    if these_keys == ["Title"]:
        subplot_titles = None  # edge case, we entered a df with no title
    else:
        subplot_titles = these_keys

    # create plot
    if len(these_keys) >= num_cols:
        graph_num_cols = num_cols
    else:
        graph_num_cols = len(these_keys)
    graph_num_rows = math.ceil((len(these_keys)) / graph_num_cols)

    this_bool = len(secondary_bars) >= 1 or secondary_y_data is not None
    specs = [
        [{"secondary_y": this_bool} for _ in range(graph_num_cols)]
        for _ in range(graph_num_rows)
    ]

    fig = sbplt.make_subplots(
        rows=graph_num_rows,
        cols=graph_num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        specs=specs,
    )

    graph_row = 0
    graph_col = 0
    for key in these_keys:
        if graph_col == graph_num_cols:
            graph_col = 0
            graph_row += 1
        for group in groups:
            vals = y_data[key].loc[:, group]
            # annotations
            if group in annotations:
                if annotation_type == "float":
                    text = list(
                        vals.astype(float).round(2).apply(lambda x: f"{x:.2f}").values
                    )
                if annotation_type == "int":
                    text = list(vals.astype(int))
                for t in range(len(text)):
                    if y_data[key].loc[idxs[t], group] < annotation_thresh:
                        text[t] = ""
            else:
                text = None

            stack_name = bar_mode_dict[group]
            offset_x = [
                x + (stack_positions[stack_name] - (num_stacks - 1) / 2) * dx
                for x in tick_vals
            ]

            for i, row in enumerate(idxs):
                if y_error_up_data is None:
                    dh = None
                else:
                    dh = [y_error_up_data[key].loc[row, group]]

                if y_error_down_data is None:
                    dl = None
                else:
                    dl = [y_error_down_data[key].loc[row, group]]

                fig.add_trace(
                    go.Bar(
                        x=[offset_x[i]],
                        y=[y_data[key].loc[row, group]],
                        error_y=dict(
                            type="data",
                            array=dh,
                            arrayminus=dl,
                            width=whisker_len,
                            thickness=error_thickness,
                        ),
                        width=dx * (1 - bar_group_gap),
                        marker_color=bar_legend["color"][group],
                        opacity=bar_legend["opacity"][group],
                        marker_pattern=dict(shape=bar_legend["pattern"][group]),
                        showlegend=False,
                        text=text[i] if group in annotations else None,
                        textfont=dict(size=annotation_size, color=annotation_color),
                        textangle=annotation_angle,
                        offsetgroup=stack_name,
                    ),
                    secondary_y=row in secondary_bars,
                    row=graph_row + 1,
                    col=graph_col + 1,
                )

            if (secondary_y_data is not None) and (
                group in secondary_y_data[key].columns
            ):
                fig.add_trace(
                    go.Scatter(
                        x=tick_vals,
                        y=secondary_y_data[key].loc[:, group],
                        mode="lines+markers",
                        marker_color=bar_legend["color"][group],
                        showlegend=False,
                        line=dict(width=line_width),
                        marker_size=marker_size,
                        connectgaps=True,
                    ),
                    secondary_y=True,
                    row=graph_row + 1,
                    col=graph_col + 1,
                )

        graph_col += 1

    if pattern_legend is not None:
        if legend_order == "reversed":
            conditions = list(reversed(pattern_legend))
        else:
            conditions = list(pattern_legend.keys())
        for condition in conditions:
            fig.add_trace(
                go.Bar(
                    x=[np.nan],
                    y=[np.nan],
                    marker_color="black",
                    marker_pattern=dict(shape=pattern_legend[condition][0]),
                    opacity=pattern_legend[condition][1],
                    name=condition,
                    showlegend=True,
                )
            )

    # legend
    if legend_order == "reversed":
        groups = list(reversed(groups))
    for group in groups:
        if bar_legend["name"][group] not in dont_add_to_legend:
            if "opacity" in bar_legend and group in bar_legend["opacity"]:
                o = bar_legend["opacity"][group]
            else:
                o = 1

            fig.add_trace(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="markers",
                    marker=dict(
                        size=legend_marker_size,
                        color=bar_legend["color"][group],
                        opacity=o,
                        symbol="square",
                    ),  # adjust size here
                    name=bar_legend["name"][group],
                    showlegend=True,
                )
            )

    # axes
    if y_range is None and force_same_yaxes:
        min_y_value, max_y_value = find_bar_min_max(y_data=y_data, bar_mode=bar_mode)
        y_range = [
            adjust_axis_bound("min", min_y_value),
            adjust_axis_bound("max", max_y_value),
        ]

    fig = fig.update_layout(
        bargap=bar_gap,
        bargroupgap=bar_group_gap,
        barmode="stack",
    )
    fig = fig.update_xaxes(tickvals=tick_vals, ticktext=idxs)

    fig = update_fig_formatting(
        fig,
        width=width * graph_num_cols,
        height=height * graph_num_rows,
        title_size=title_size,
        text_size=text_size,
        legend_size=legend_size,
        y_axis_title=y_axis_title,
        x_axis_title=x_axis_title,
        y_zerolinecolor=y_zerolinecolor,
        x_zerolinecolor=x_zerolinecolor,
        grid_color=grid_color,
        y_range=y_range,
    )
    # secondary axis
    if len(secondary_bars) >= 1:
        secondary_y_data = {}
        for project in y_data:
            secondary_y_data[project] = y_data[project].loc[secondary_bars, :]

    if secondary_y_data is not None:
        fig.update_yaxes(
            title_text=secondary_y_axis_title,
            secondary_y=True,
        )
        if secondary_y_range is None and force_same_yaxes:
            min_y_value, max_y_value = find_bar_min_max(
                y_data=secondary_y_data, bar_mode=bar_mode
            )
            secondary_y_range = [
                adjust_axis_bound("min", min_y_value),
                adjust_axis_bound("max", max_y_value),
            ]
        fig.update_yaxes(range=secondary_y_range, secondary_y=True)
    return fig
