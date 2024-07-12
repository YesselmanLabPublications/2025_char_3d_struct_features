import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from typing import List, Union, Optional

from rna_secstruct_design.selection import get_selection, SecStruct

from dms_3d_features.logger import get_logger

log = get_logger("plotting")


def colors_for_sequence(seq: str) -> list:
    """
    Returns a list of colors corresponding to the input DNA/RNA sequence.

    This function maps each character in the input DNA/RNA sequence to a specific color:
    - 'A' -> 'red'
    - 'C' -> 'blue'
    - 'G' -> 'orange'
    - 'T' -> 'green'
    - 'U' -> 'green'

    Args:
        seq (str): A string representing a RNA/DNA sequence. The sequence is
                   expected to contain only the characters 'A', 'C', 'G', 'U', and 'T'.

    Returns:
        list: A list of strings where each element is a color corresponding
              to a character in the DNA sequence.

    Raises:
        ValueError: If the sequence contains characters other than 'A', 'C',
                    'G', 'U' and 'T'.

    Example:
        >>> colors_for_sequence("ACGT")
        ['red', 'blue', 'orange', 'green']

    """
    color_mapping = {"A": "red", "C": "blue", "G": "orange", "T": "green", "U": "green"}
    colors = []
    for e in seq:
        try:
            color = color_mapping[e]
            colors.append(color)
        except KeyError as exc:
            log.error(
                f"Invalid character {e} in sequence. Sequence must contain only 'A', 'C', 'G', 'U', and 'T'."
            )
            raise ValueError(f"Invalid character {e} in sequence.") from exc

    log.debug(f"Input Sequence: {seq}")
    log.debug(f"Output Colors: {colors}")
    return colors


def find_stretches(nums: List[int]) -> List[List[int]]:
    """Finds all consecutive number stretches in a list of integers.

    Args:
        nums (List[int]): A list of integers that may contain consecutive numbers.

    Returns:
        List[List[int]]: A list of lists, each containing the start and end of a consecutive number stretch.

    Raises:
        ValueError: If `nums` contains non-integer elements.

    Example:
        >>> find_stretches([3, 4, 5, 10, 11, 12])
        [[3, 5], [10, 12]]

        >>> find_stretches([1, 2, 3, 7, 8, 10])
        [[1, 3], [7, 8], [10, 10]]

    Notes:
        The input list is sorted within the function to simplify the logic for finding consecutive stretches.
    """

    log.debug("Initial list: %s", nums)

    if len(nums) == 0:
        return []

    nums = sorted(set(nums))
    log.debug("Sorted and de-duplicated list: %s", nums)

    stretches = []
    start = end = nums[0]

    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            stretches.append([start, end])
            start = end = num

    stretches.append([start, end])
    log.debug("Identified stretches: %s", stretches)
    return stretches


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


def trim(
    content: Union[str, list], prime_5: int, prime_3: int
) -> Union[str, list, str]:
    """
    Trims a string or list from the 5' (start) and 3' (end) ends.

    Args:
        content: The content to be trimmed, can be a string or list.
        prime_5: The number of elements to trim from the 5' end (start).
        prime_3: The number of elements to trim from the 3' end (end).

    Returns:
        A trimmed string or list, or an error message if the content type is invalid.

    Raises:
        TypeError: If the content is neither a string nor a list.
    """
    if not isinstance(content, (str, list)):
        log.error("Invalid content type. Please provide a string or a list.")
        raise TypeError("Invalid content type. Please provide a string or a list.")

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
            "sequence_structure", "sequence", or "structure". Defaults to "sequence_structure".
        trim_5p: The number of nucleotides to trim from the 5' end.
            Defaults to 0.
        trim_3p: The number of nucleotides to trim from the 3' end.
            Defaults to 0.
        highlights: List of highlight regions. Each highlight
            region should be a tuple of start and end indices. Defaults to None.

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
        row (pandas.Series): The row of data containing the sequence, structure, and data columns.
        data_col (str, optional): The name of the column containing the data. Defaults to "data".
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    return plot_pop_avg(row["sequence"], row["structure"], row[data_col], ax)


def plot_pop_avg_all(df, data_col="data", axis="sequence_structure", **kwargs):
    """
    Plots the population average for each row in the given DataFrame. plots are seperated
    and are in a column format.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        data_col (str, optional): The column name in the DataFrame that contains the data to be plotted. Defaults to "data".
        axis (str, optional): The axis along which to calculate the population average. Defaults to "sequence_structure".
        **kwargs: Additional keyword arguments to be passed to the plt.subplots() function.

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
        titration_col (str): The name of the column in `df` representing the titration values.
        highlights (list, optional): A list of values to highlight in the plot. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the `subplots` function.

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
        axes[j].set_title(str(row[titration_col]) + " mM")
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


# style functions #############################################################


def publication_style_ax(ax):
    """
    Sets the publication style for the given matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to apply the publication style to.

    Returns:
        None
    """
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    fsize = 24
    ax.xaxis.label.set_fontsize(fsize)
    ax.yaxis.label.set_fontsize(fsize)
    ax.tick_params(axis="both", which="major", labelsize=fsize - 2)


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
