import pandas as pd
from tabulate import tabulate


def dataframe_column_table(df, num_columns=4):
    """
    Create a nicely formatted table of all columns in multiple columns.

    Args:
    df (pd.DataFrame): The DataFrame whose columns to display.
    num_columns (int): Number of columns to display the data in.

    Returns:
    None: Prints the formatted table.
    """
    column_data = pd.DataFrame({"Column Name": df.columns})
    column_data["Column Index"] = range(len(column_data))
    column_data["Display Column"] = column_data["Column Index"] % num_columns
    column_data["Display Row"] = column_data["Column Index"] // num_columns

    pivot_data = column_data.pivot(
        index="Display Row", columns="Display Column", values="Column Name"
    )
    pivot_data.columns.name = None
    pivot_data.index.name = None
    # Format the table as a string
    table_str = pivot_data.to_string(index=False)
    # Print the formatted table
    print(table_str)


def generate_threshold_summary(
    df, y_column, threshold=-5.45, greater_than=False, sort=True
):
    """Generates a summary table based on a threshold comparison for a specific column.

    This function creates a summary table that shows the percentage of values in a specified
    column that are above or below a given threshold, along with the count of occurrences for
    each unique value in the y_column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        y_column (str): The name of the column to group by and summarize.
        threshold (float, optional): The threshold value for comparison. Defaults to -5.45.
        greater_than (bool, optional): If True, calculates percentage above threshold.
            If False, calculates percentage below threshold. Defaults to False.
        sort (bool, optional): If True, sorts the summary by percentage in descending order.
            Defaults to True.

    Returns:
        None: Prints the formatted summary table to the console.
    """
    # Calculate percentages and counts
    summary = []
    comparison = ">" if greater_than else "<"
    for y_value in df[y_column].unique():
        group = df[df[y_column] == y_value]
        if greater_than:
            percent = (group["ln_r_data"] > threshold).mean() * 100
        else:
            percent = (group["ln_r_data"] < threshold).mean() * 100
        count = len(group)
        summary.append([y_value, f"{percent:.2f}%", count])
    # Sort by percentage descending if sort is True
    if sort:
        summary.sort(key=lambda x: float(x[1][:-1]), reverse=True)
    # Create table
    headers = [y_column, f"% {comparison} {threshold}", "Count"]
    table = tabulate(summary, headers=headers, tablefmt="pipe", floatfmt=".2f")
    print(f"Summary table for {y_column}:")
    print(table)
