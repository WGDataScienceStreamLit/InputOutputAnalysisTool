import numpy as np
import pandas as pd
import plotly.graph_objects as go


# function to edit a df to make it display friendly
def create_display_df(df: pd.DataFrame, names_col: str) -> pd.DataFrame:
    """
    Create a display version of a dataframe that simplifies the output for UI

    :param df: Dataframe containing the IO data to display
    :param names_col: Name of the col that contains sector names
    :returns: Dataframe with the names col as index
    """

    # set up a display dataframe
    display_df = df.copy()

    display_df.index = display_df[names_col]
    display_df.drop(names_col, axis=1, inplace=True)

    return display_df


# convert the df to csv (for download edit)
def convert_df(df: pd.DataFrame) -> str:
    """
    For dataframe downloads, encode as utf-8 str.

    :param df: Pandas DF to download
    :returns: DF CSV as utf-8 str
    """

    return df.to_csv().encode("utf-8")


def model_bar(
    direct_demand: np.array,
    indirect_demand: np.array,
    induced_demand: np.array,
    sectors: list,
    modelled_demand: np.array,
    incind: bool,
) -> go.Figure:
    """
    Bar plot displaying the modelled scenario

    :param direct_demand: Vector
    :param indirect_demand: Vector
    :param induced_demand: Vector
    :param sectors: List of sectors
    :param modelled_demand: Vector
    :param incind: Bool (include induced effects)
    :return fig: Figure
    """

    # Build a plot based on the new demands
    demand_types = [direct_demand, indirect_demand, induced_demand]
    demand_labels = ["Direct Demand", "Indirect Demand", "Induced Demand"]
    colors = ["#003D6B", "#5D99C6", "#071527"]

    fig = go.Figure()

    for i, demand in enumerate(demand_types):
        if demand is not None:
            # Only view sectors with changes (ends up being all)
            demand = demand[modelled_demand > 0]
            sectors_vis = [s for s, d in zip(sectors, modelled_demand) if d > 0]

            if incind:
                sectors_vis.append("Households")

            fig.add_trace(
                go.Bar(
                    y=sectors_vis,
                    x=demand,
                    customdata=sectors_vis,
                    hovertemplate="Sector: %{customdata} <br>Value: %{x}",
                    name=demand_labels[i],
                    orientation="h",
                    marker=dict(color=colors[i]),
                )
            )

    return fig


def comp_bar(
    direct_demand: np.array,
    indirect_demand: np.array,
    induced_demand: np.array,
    sectors: list,
    output: np.array,
    incind: bool,
    filters: list,
) -> go.Figure:
    """
    Build the plot comparing modelled scenario to current

    :param direct_demand: Vector
    :param indirect_demand: Vector
    :param induced_demand: Vector
    :param sectors: List of sectors
    :param output: Vector
    :param incind: Bool (include induced effects)
    :param filter: List of sectors to display
    :return fig: Figure
    """

    fig = go.Figure()

    # Filter the demand by sectors
    filter_indices = [sectors.index(f) for f in filters]

    filtered_output = output[filter_indices]
    filtered_direct_demand = direct_demand[filter_indices]
    filtered_indirect_demand = indirect_demand[filter_indices]

    # Get the demand types
    points = [filtered_output, filtered_direct_demand, filtered_indirect_demand, None]
    point_labels = [
        "Current Demand",
        "Direct Demand",
        "Indirect Demand",
        "Induced Demand",
    ]
    point_colors = ["#2E8B57", "#003D6B", "#5D99C6", "#071527"]

    if incind:
        filtered_induced_demand = induced_demand[filter_indices]
        points[-1] = filtered_induced_demand

    for i, p in enumerate(points):
        if p is not None:
            fig.add_trace(
                go.Bar(
                    y=filters,
                    x=p,
                    customdata=filters,
                    hovertemplate="Sector: %{customdata} <br>Value: %{x}",
                    name=point_labels[i],
                    orientation="h",
                    marker=dict(color=point_colors[i]),
                )
            )

    return fig


def update_plots(plot: go.Figure, title: str, height: int = 1980) -> go.Figure:
    """
    Updates a plot to add styling, resize and add title.

    :param plot: The plotly figure
    :param title: Title to add to the figue
    :param height: Height to plot figure at (width not included as done in streamlit dynamically)
    :returns: Plotly figure with new style, title and height
    """
    # update layout to sensible size, legend order and stack the bars
    plot.update_layout(
        template="seaborn",
        height=height,
        yaxis=dict(type="category", categoryorder="total ascending"),
        barmode="stack",
        legend={"traceorder": "normal"},
        title=title,
    )

    return plot


def display_text(path: str) -> str:
    """
    Read text from a txt file

    :param path: Path to the txt file
    :returns: Text from the file
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    return content
