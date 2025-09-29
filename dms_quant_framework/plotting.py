import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Union, Optional
from scipy.stats import ks_2samp, pearsonr, linregress
from sklearn.linear_model import LinearRegression

from rna_secstruct_design.selection import get_selection, SecStruct

from dms_quant_framework.logger import get_logger
from dms_quant_framework.util import find_stretches

log = get_logger("plotting")


def colors_for_sequence(seq: str) -> List[str]:
    """
    Returns a list of colors corresponding to the input DNA/RNA sequence.

    This function maps each character in the input DNA/RNA sequence to a specific color:
    - 'A' -> 'red'
    - 'C' -> 'blue'
    - 'G' -> 'orange'
    - 'T' or 'U' -> 'green'
    - '&' -> 'gray'

    Args:
        seq (str): A string representing a RNA/DNA sequence.

    Returns:
        List[str]: A list of color strings corresponding to each character in the
        sequence.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the sequence contains invalid characters.
    """
    color_mapping = {
        "A": "red",
        "C": "blue",
        "G": "orange",
        "T": "green",
        "U": "green",
        "&": "gray",
    }

    colors = []
    for e in seq.upper():  # Convert to uppercase
        try:
            color = color_mapping[e]
            colors.append(color)
        except KeyError as exc:
            log.error(
                "Invalid character '{}' in sequence. Sequence must contain only "
                "'A', 'C', 'G', 'U', 'T', and '&'.".format(e)
            )
            raise ValueError("Invalid character '{}' in sequence.".format(e)) from exc

    log.debug("Input Sequence: {}".format(seq))
    log.debug("Output Colors: {}".format(colors))
    return colors


def fill_between(
    ax: axes.Axes,
    color: str,
    x: List[float],
    y: List[float],
    alpha: float = 0.15,
    **kwargs,
) -> None:
    """
    Fills the area between two curves on the given axes.

    Args:
        ax: The axes on which to plot.
        color: The color of the filled area.
        x: The x-coordinates of the curves defining the area.
        y: The y-coordinates of the curves defining the area.
        alpha: The transparency of the filled area. Default is 0.15.
        **kwargs: Additional keyword arguments to be passed to the `fill_between` function.

    Returns:
        None
    """
    ax.fill_between(x, y, color=color, alpha=alpha, zorder=-1)


def trim(content: Union[str, list], prime_5: int, prime_3: int) -> Union[str, list]:
    """
    Trims a string or list from the 5' (start) and 3' (end) ends.

    Args:
        content: The content to be trimmed, can be a string or list.
        prime_5: The number of elements to trim from the 5' end (start).
        prime_3: The number of elements to trim from the 3' end (end).

    Returns:
        A trimmed string or list.

    """
    trimmed_content = content[prime_5 : -prime_3 or None]
    log.debug(f"Trimmed content: {trimmed_content}")
    return trimmed_content


def plot_pop_avg(
    seq: str,
    ss: str,
    reactivities: List[float],
    ax: Optional[plt.Axes] = None,
    axis: str = "sequence_structure",
    trim_5p: int = 0,
    trim_3p: int = 0,
    highlights: Optional[List[tuple]] = None,
) -> plt.Axes:
    """
    Plot DMS reactivity for a sequence and secondary structure.

    Args:
        seq: The sequence.
        ss: The secondary structure.
        reactivities: List of reactivities.
        ax: The matplotlib axis to plot on. If not provided,
            a new figure and axis will be created.
        axis: The axis to plot on. Possible values are
            "sequence_structure", "sequence", or "structure".
            Defaults to "sequence_structure".
        trim_5p: The number of nucleotides to trim from the 5' end.
            Defaults to 0.
        trim_3p: The number of nucleotides to trim from the 3' end.
            Defaults to 0.
        highlights: List of highlight regions. Each highlight
            region should be a tuple of start and end indices.
            Defaults to None.

    Returns:
        The plotted axis.

    """
    seq = trim(seq, trim_5p, trim_3p)
    ss = trim(ss, trim_5p, trim_3p)
    reactivities = trim(reactivities, trim_5p, trim_3p)
    highlight_bounds = []
    if highlights is None:
        highlights = []
    for h in highlights:
        selection = get_selection(SecStruct(seq, ss), h)
        for bounds in find_stretches(selection):
            highlight_bounds.append(bounds)
    colors = colors_for_sequence(seq)
    x = list(range(len(seq)))
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 4))
    ax.bar(range(0, len(reactivities)), reactivities, color=colors)
    ax.set_xticks(x)
    for bounds in highlight_bounds:
        fill_between(ax, "gray", bounds, [0, 10])
    if axis == "sequence_structure":
        ax.set_xticklabels([f"{s}\n{nt}" for s, nt in zip(seq, ss)])
    elif axis == "sequence":
        ax.set_xticklabels([f"{s}" for s in seq])
    elif axis == "structure":
        ax.set_xticklabels([f"{s}" for s in ss])
    else:
        pass
    return ax


def plot_pop_avg_from_row(row, data_col="data", ax=None):
    """
    Plots the population average from a given row of data.

    Args:
        row (pandas.Series): The row of data containing the sequence, structure,
            and data columns.
        data_col (str, optional): The name of the column containing the data.
            Defaults to "data".
        ax (matplotlib.axes.Axes, optional): The axes on which to plot.
            Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    return plot_pop_avg(row["sequence"], row["structure"], row[data_col], ax)


def plot_pop_avg_all(df, data_col="data", axis="sequence_structure", **kwargs):
    """
    Plots the population average for each row in the given DataFrame. plots are
    seperated and are in a column format.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        data_col (str, optional): The column name in the DataFrame that contains
            the data to be plotted. Defaults to "data".
        axis (str, optional): The axis along which to calculate the population
            average. Defaults to "sequence_structure".
        **kwargs: Additional keyword arguments to be passed to the plt.subplots()
            function.

    Returns:
        matplotlib.figure.Figure: The generated figure object.

    """
    fig, axes = plt.subplots(len(df), 1, **kwargs)
    j = 0
    for i, row in df.iterrows():
        colors = colors_for_sequence(row["sequence"])
        axes[j].bar(range(0, len(row[data_col])), row[data_col], color=colors)
        axes[j].set_title(row["rna_name"])
        j += 1
    plot_pop_avg_from_row(df.iloc[-1], ax=axes[-1], axis=axis)
    return fig


def plot_pop_avg_titration(df, titration_col, highlights=None, **kwargs):
    """
    Plots the population average titration for a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        titration_col (str): The name of the column in `df` representing the
            titration values.
        highlights (list, optional): A list of values to highlight in the plot.
            Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the `subplots`
            function.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """
    fig, axes = plt.subplots(len(df), 1, **kwargs)
    j = 0
    secstruct = SecStruct(df.iloc[0]["sequence"], df.iloc[0]["structure"])
    highlight_bounds = []
    if highlights is None:
        highlights = []
    else:
        for h in highlights:
            selection = get_selection(secstruct, h)
            for bounds in find_stretches(selection):
                highlight_bounds.append(bounds)
    for i, row in df.iterrows():
        colors = colors_for_sequence(row["sequence"])
        axes[j].bar(range(0, len(row["data"])), row["data"], color=colors)
        axes[j].set_title(f"{row[titration_col]} mM")
        axes[j].set_ylim([0, 0.1])
        axes[j].set_xlim([-0.1, len(row["data"]) + 0.1])
        axes[j].set_xticks([])
        for bounds in highlight_bounds:
            fill_between(axes[j], "gray", bounds, [0, 10])
        j += 1
    plot_pop_avg_from_row(df.iloc[-1], ax=axes[-1])
    return fig


def plot_pop_avg_traces_all(df, plot_sequence=False, ylim=None, **kwargs):
    """
    Plots population average traces for all data points in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data points.
        plot_sequence (bool, optional): Whether to plot the sequence information. Defaults to False.
        ylim (float, optional): The y-axis limit for the plot. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the plt.subplots() function.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """
    fig, ax = plt.subplots(1, 1, **kwargs)
    for i, row in df.iterrows():
        if "label" in row:
            label = row["label"]
        else:
            label = row["rna_name"]
        plt.plot(row["data"], label=label, lw=4)
    # fig.legend(loc="upper left")
    if plot_sequence:
        seq = df.iloc[0]["sequence"]
        ss = df.iloc[0]["structure"]
        x = list(range(len(seq)))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{nt}" for s, nt in zip(seq, ss)])
    if ylim is not None:
        ax.set_ylim([0, ylim])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    return fig


def plot_motif_boxplot_stripplot(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None, show_structure: bool = False
) -> plt.Axes:
    """
    Plots a boxplot and stripplot for motif data.

    Args:
        df (pd.DataFrame): The input dataframe containing the motif data.
        ax (Optional[plt.Axes]): The matplotlib axes object to plot on. If not provided, a new figure and axes will be created.
        show_structure (bool): Whether to show the structure information in labels. Defaults to False.

    Returns:
        plt.Axes: The matplotlib axes object containing the plot.

    Raises:
        ValueError: If the required columns are missing in the dataframe.
    """
    required_columns = ["m_sequence", "m_structure", "r_loc_pos", "r_data"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    sequence = df.iloc[0]["m_sequence"]
    structure = df.iloc[0]["m_structure"]
    positions = list(range(len(sequence)))
    colors = colors_for_sequence(sequence)
    custom_palette = dict(zip(positions, colors))

    labels = [f"{n}\n{s}" if show_structure else n for n, s in zip(sequence, structure)]

    sns.boxplot(
        x="r_loc_pos",
        y="r_data",
        data=df,
        order=positions,
        hue="r_loc_pos",
        palette=custom_palette,
        showfliers=False,
        linewidth=0.5,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        x="r_loc_pos",
        y="r_data",
        data=df,
        order=positions,
        color="black",
        size=1,
        ax=ax,
    )

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nucleotide", labelpad=2)
    ax.set_ylabel("Mutation Fraction", labelpad=2)

    return ax


def plot_whole_pdb_reactivity(df: pd.DataFrame, ax=None) -> plt.Axes:
    """
    Plots the whole RNA data points.

    Args:
        df_sub (pd.DataFrame): A DataFrame
        ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib axis to plot on.
            If None, a new axis is created. Defaults to None.

    Returns:
        axes: The matplotlib axis object with the plotted data.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    sequence = df.iloc[0]["m_sequence"]
    pos = list(range(len(sequence)))

    labels = []
    for n, s in zip(df.iloc[0]["m_sequence"], df.iloc[0]["m_structure"]):
        labels.append(f"{n}\n{s}")

    sns.scatterplot(
        x="r_loc_pos",
        y="whole_rna_reac",
        data=df,
        color="magenta",
        s=70,
        ax=ax,
        zorder=3,
    )
    ax.set_xticks(ticks=range(len(pos)), labels=labels)
    return ax


def plot_violins_w_percent(
    df: pd.DataFrame,
    x: str,
    y: str,
    cutoff=-5.45,
    color="tab:gray",
    colors=None,
    gt_lt="greater",
    text_offset=2.5,
    xlim=None,
    ax=None,
    sorted_by_mean: bool = False,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.0, 1.5), dpi=200)

    # Calculate percentages
    def calculate_percentage(group):
        condition = (
            group["ln_r_data"] > cutoff
            if gt_lt == "greater"
            else group["ln_r_data"] < cutoff
        )
        return (condition.mean() * 100).round(2)

    percentages = df.groupby(y).apply(calculate_percentage)
    if sorted_by_mean:
        percentages = percentages.sort_values(ascending=False)

    if colors is not None:
        palette = dict(zip(percentages.index, colors))
        sns.violinplot(
            x=x,
            y=y,
            data=df,
            palette=palette,
            density_norm="width",
            ax=ax,
            linewidth=0.5,
            order=percentages.index,
            legend=False,
        )
    else:
        sns.violinplot(
            x=x,
            y=y,
            data=df,
            hue=y if color is None else None,
            palette=sns.color_palette() if color is None else None,
            color=color,
            density_norm="width",
            ax=ax,
            linewidth=0.5,
            order=percentages.index,
            legend=False,
        )
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    # Add percentage labels
    for i, y_value in enumerate(percentages.index):
        ax.text(
            ax.get_xlim()[0] + text_offset,
            i + 0.03,
            f"{percentages[y_value]:.2f}%",
            va="center",
            ha="right",
        )
    return ax


def plot_violins_w_percent_groups(
    df,
    x_col,
    y_col,
    gt_lt="greater",
    color="tab:gray",
    n_panels=2,
    xlim=(-10, 1),
    figsize=(2.0, 1.5),
    dpi=200,
    sorted_by_mean: bool = False,
):
    df = df.copy()
    df.sort_values(y_col, inplace=True, ascending=True)

    # Split groups into n_panels
    groups = df[y_col].unique()
    group_splits = np.array_split(groups, n_panels)
    scale_factor = 1.35
    # Create subplots
    fig, axes = plt.subplots(
        1, n_panels, figsize=(figsize[0] * n_panels * scale_factor, figsize[1]), dpi=dpi
    )
    if n_panels == 1:
        axes = [axes]

    for i, (ax, group_split) in enumerate(zip(axes, group_splits)):
        df_subset = df[df[y_col].isin(group_split)]
        plot_violins_w_percent(
            df=df_subset,
            x=x_col,
            y=y_col,
            cutoff=-5.45,
            color=color,
            gt_lt=gt_lt,
            text_offset=1.5 * scale_factor,
            xlim=xlim,
            sorted_by_mean=sorted_by_mean,
            ax=ax,
        )
        ax.set_ylabel("Motif Topology", labelpad=2)
        ax.set_xlabel("ln(Mutation Fraction)", labelpad=2)
        ax.set_xticks([-10, -8, -6, -4, -2])
        format_small_plot(ax)
    return fig, axes


def plot_scatter_w_best_fit_line(x, y, size=1, ax=None):
    """
    Plots a scatter plot and the best fit line for the given data, including the R^2 value.

    Args:
        x (array-like): The x-values of the data points.
        y (array-like): The y-values of the data points.
        ax (matplotlib.axes.Axes, optional): The Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.axes.Axes: The Axes object with the scatter plot and best fit line.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(x, y, s=size)

    # Reshape x for sklearn
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # Linear regression model
    model = LinearRegression()
    model.fit(x, y)

    # Get the slope and intercept of the line
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate x values for the best fit line
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept

    # Plot the best fit line
    ax.plot(x_fit, y_fit, color="black", label="Best fit line", lw=1)

    # Calculate Pearson correlation coefficient and R^2
    r, _ = pearsonr(x.flatten(), y)
    r_squared = r**2

    # Add R^2 annotation
    ax.text(
        0.03,
        0.97,
        f"$R^2 = {r_squared:.2f}$",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    return ax


# style functions #############################################################


def publication_style_ax(
    ax, fsize: int = 10, ytick_size: int = 8, xtick_size: int = 8
) -> None:
    """
    Applies publication style formatting to the given matplotlib Axes object.
    Args:
        ax (matplotlib.axes.Axes): The Axes object to apply the formatting to.
        fsize (int, optional): The font size for labels, title, and tick labels. Defaults to 10.
        ytick_size (int, optional): The font size for y-axis tick labels. Defaults to 8.
        xtick_size (int, optional): The font size for x-axis tick labels. Defaults to 8.
    Returns:
        None
    """
    # Set line widths and tick widths
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, size=1.5, pad=1)

    # Set font sizes for labels and title
    ax.xaxis.label.set_fontsize(fsize)
    ax.yaxis.label.set_fontsize(fsize)
    ax.title.set_fontsize(fsize)

    # Set font names for labels, title, and tick labels
    ax.xaxis.label.set_fontname("Arial")
    ax.yaxis.label.set_fontname("Arial")
    ax.title.set_fontname("Arial")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    # Set font sizes for tick labels
    for label in ax.get_yticklabels():
        label.set_fontsize(ytick_size)
    for label in ax.get_xticklabels():
        label.set_fontsize(xtick_size)

    # Set font sizes for text objects added with ax.text()
    for text in ax.texts:
        text.set_fontname("Arial")
        text.set_fontsize(fsize - 2)


def publication_scatter(ax, x, y, **kwargs):
    """
    Scatter plot for publication.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the scatter plot on.
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        **kwargs: Additional keyword arguments to be passed to the `scatter` function.

    Returns:
        None
    """
    ax.scatter(x, y, s=150, **kwargs)


def publication_line(ax, x, y, **kwargs):
    """
    Plots a line on the given axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        x (array-like): The x-coordinates of the line.
        y (array-like): The y-coordinates of the line.
        **kwargs: Additional keyword arguments to pass to the `plot` function.

    Returns:
        None
    """
    ax.plot(x, y, markersize=10, lw=2, **kwargs)


def format_small_plot(ax):
    """
    Formats a small plot with specified style parameters. Plot is expected to have
    a single subplot and setup like

    ```python
    fig, ax = plt.subplots(figsize=(1.50, 1.25), dpi=200)
    ```

    Args:
        ax: The matplotlib Axes object to format.

    Returns:
        None
    """
    publication_style_ax(ax, fsize=8, ytick_size=6, xtick_size=6)
    plt.subplots_adjust(left=0.3, bottom=0.21, top=0.98)


def calculate_subplot_coordinates(
    fig_size_inches, subplot_layout, subplot_size_inches, spacing=None
):
    """
    Calculate subplot coordinates for matplotlib subplots with exact subplot sizes.

    Parameters:
    -----------
    fig_size_inches : tuple
        Figure size as (width, height) in inches
    subplot_layout : tuple
        Number of subplots as (rows, columns)
    subplot_size_inches : tuple
        Exact size of each subplot as (width, height) in inches
    spacing : dict, optional
        Spacing parameters as {'hspace': float, 'wspace': float, 'margins': dict}
        - hspace: horizontal spacing between subplots in inches
        - wspace: vertical spacing between subplots in inches
        - margins: {'left': float, 'right': float, 'top': float, 'bottom': float} in inches
        Defaults to 0.5 inch spacing and 0.75 inch margins

    Returns:
    --------
    list
        List of tuples, each containing (left, bottom, width, height) coordinates
        in figure-relative units (0-1) for each subplot

    Raises:
    -------
    Warning if subplots won't fit in the specified figure size
    """
    import warnings

    # Default spacing if not provided
    if spacing is None:
        spacing = {
            "hspace": 0.5,  # horizontal spacing in inches
            "wspace": 0.5,  # vertical spacing in inches
            "margins": {"left": 0.75, "right": 0.75, "top": 0.75, "bottom": 0.75},
        }

    fig_width, fig_height = fig_size_inches
    rows, cols = subplot_layout
    subplot_width, subplot_height = subplot_size_inches

    # Calculate total space needed
    total_subplot_width = cols * subplot_width
    total_subplot_height = rows * subplot_height

    total_hspace = (cols - 1) * spacing["hspace"]
    total_wspace = (rows - 1) * spacing["wspace"]

    total_margins_width = spacing["margins"]["left"] + spacing["margins"]["right"]
    total_margins_height = spacing["margins"]["top"] + spacing["margins"]["bottom"]

    required_width = total_subplot_width + total_hspace + total_margins_width
    required_height = total_subplot_height + total_wspace + total_margins_height

    # Check if subplots fit and issue warnings
    if required_width > fig_width:
        warnings.warn(
            f"Subplots won't fit horizontally! Required width: {required_width:.2f}\", "
            f'Figure width: {fig_width:.2f}". Consider increasing figure width or '
            f"reducing subplot width/spacing."
        )

    if required_height > fig_height:
        warnings.warn(
            f"Subplots won't fit vertically! Required height: {required_height:.2f}\", "
            f'Figure height: {fig_height:.2f}". Consider increasing figure height or '
            f"reducing subplot height/spacing."
        )

    # Calculate coordinates even if they don't fit (for debugging)
    coordinates = []

    # Convert inches to figure-relative coordinates
    for row in range(rows):
        for col in range(cols):
            # Calculate position in inches
            left_inches = spacing["margins"]["left"] + col * (
                subplot_width + spacing["hspace"]
            )
            bottom_inches = spacing["margins"]["bottom"] + (rows - 1 - row) * (
                subplot_height + spacing["wspace"]
            )

            # Convert to figure-relative coordinates (0-1)
            left_rel = left_inches / fig_width
            bottom_rel = bottom_inches / fig_height
            width_rel = subplot_width / fig_width
            height_rel = subplot_height / fig_height

            coordinates.append((left_rel, bottom_rel, width_rel, height_rel))

    return coordinates


def merge_neighboring_coords(coords_list, indices):
    """
    Merge a list of coordinate positions into one bigger coordinate.
    The indices do not need to be consecutive in the list, but the corresponding
    rectangles must be adjacent (touching) in the figure.

    Parameters:
    -----------
    coords_list : list
        List of coordinate tuples (left, bottom, width, height)
    indices : list of int
        List of indices of coordinates to merge. Must be adjacent (touching).

    Returns:
    --------
    list
        Updated list with merged coordinates
    """
    if not indices:
        raise ValueError("indices list cannot be empty")
    if any(i < 0 or i >= len(coords_list) for i in indices):
        raise ValueError("indices out of range")

    # Get the coordinates to merge
    coords_to_merge = [coords_list[i] for i in indices]

    # Check that all rectangles are touching (adjacent)
    # We'll use a simple approach: for each pair, check if they touch
    def rects_touch(r1, r2, tol=1e-8):
        # r = (left, bottom, width, height)
        l1, b1, w1, h1 = r1
        l2, b2, w2, h2 = r2
        r1_right = l1 + w1
        r1_top = b1 + h1
        r2_right = l2 + w2
        r2_top = b2 + h2

        # Check for overlap or touching on any edge
        horizontal_touch = (
            abs(r1_right - l2) < tol
            or abs(r2_right - l1) < tol
            or (l1 < r2_right - tol and r1_right > l2 + tol)
        )
        vertical_overlap = (b1 < r2_top - tol and b1 + h1 > b2 + tol) or (
            b2 < r1_top - tol and b2 + h2 > b1 + tol
        )
        vertical_touch = (
            abs(r1_top - b2) < tol
            or abs(r2_top - b1) < tol
            or (b1 < r2_top - tol and r1_top > b2 + tol)
        )
        horizontal_overlap = (l1 < r2_right - tol and l1 + w1 > l2 + tol) or (
            l2 < r1_right - tol and l2 + w2 > l1 + tol
        )
        # Touching if they share an edge (either horizontally or vertically)
        return (horizontal_touch and vertical_overlap) or (
            vertical_touch and horizontal_overlap
        )

    # Calculate the merged coordinates
    left = min(coord[0] for coord in coords_to_merge)
    bottom = min(coord[1] for coord in coords_to_merge)
    right = max(coord[0] + coord[2] for coord in coords_to_merge)
    top = max(coord[1] + coord[3] for coord in coords_to_merge)
    width = right - left
    height = top - bottom

    # Create new list with merged coordinates
    new_coords_list = []
    n = len(coords_list)
    idx_set = set(indices)
    merged = False
    for i in range(n):
        if i in idx_set and not merged:
            new_coords_list.append((left, bottom, width, height))
            merged = True
        elif i not in idx_set:
            new_coords_list.append(coords_list[i])
    return new_coords_list
