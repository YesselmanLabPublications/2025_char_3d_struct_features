import pandas as pd
from scipy.stats import ttest_ind, pearsonr


def r2(x, y):
    """
    Calculate the coefficient of determination (R^2) for two variables.

    Args:
    x (array-like): The first variable.
    y (array-like): The second variable.

    Returns:
    float: The R^2 value, which represents the proportion of the variance in the dependent variable
           that is predictable from the independent variable.
    """
    return pearsonr(x, y)[0] ** 2


def check_pairwise_statistical_significance(df, group_col, value_col):
    """
    This function checks for statistical significance between all pairs of grouped
    distributions.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    group_col (str): The column name to group by.
    value_col (str): The column name containing the values to compare.

    Returns:
    pd.DataFrame: A dataframe containing the p-values for each pair of groups.
    """
    groups = df.groupby(group_col)[value_col].apply(list)
    results = []

    for i, (group1, values1) in enumerate(groups.items()):
        for j, (group2, values2) in enumerate(groups.items()):
            if i < j:
                t_stat, p_val = ttest_ind(values1, values2, equal_var=False)
                results.append({"Group 1": group1, "Group 2": group2, "p-value": p_val})

    return pd.DataFrame(results)
