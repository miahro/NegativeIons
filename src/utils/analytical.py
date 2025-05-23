import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def calculate_vif(data, target_column):

    numeric_data = data.select_dtypes(
        include=[np.number])  # Select only numeric columns
    # Drop the target column
    numeric_data.drop(columns=[target_column], inplace=True)
    numeric_data.dropna(inplace=True)  # Drop rows with missing values
    vif_data = pd.DataFrame()
    vif_data["Variable"] = numeric_data.columns

    # Suppress divide by zero warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        vif_data["VIF"] = [variance_inflation_factor(
            numeric_data.values, i) for i in range(numeric_data.shape[1])]

    return vif_data


def plot_histograms(data, plots_per_row=4):
    numeric_data = data.select_dtypes(include=[np.number])
    num_columns = len(numeric_data.columns)
    num_rows = (num_columns + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(
        plots_per_row * 3.5, num_rows * 3.5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_data.columns):
        unique_values = numeric_data[column].nunique()
        numeric_data[column].hist(ax=axes[i], bins=48, edgecolor='black')
        axes[i].set_title(f"{column}\n(Unique: {unique_values})", fontsize=13)
        axes[i].set_xlabel('Value', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_histograms_by_year(annual_dfs, plots_per_row=4):
    df_all = pd.concat([df.assign(year=year)
                       for year, df in annual_dfs.items()]).reset_index(drop=True)
    numeric_columns = df_all.select_dtypes(
        include=[np.number]).columns.tolist()

    if "year" in numeric_columns:
        numeric_columns.remove("year")

    num_columns = len(numeric_columns)
    num_rows = (num_columns + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(num_rows, plots_per_row,
                             figsize=(plots_per_row * 4, num_rows * 3.5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.histplot(data=df_all, x=column, hue="year", bins=48,
                     edgecolor="black", alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{column} Distribution per Year", fontsize=13)
        axes[i].set_xlabel("Value", fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                cmap='coolwarm', cbar=True, annot_kws={'size': 11})
    plt.title('Correlation Matrix', fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()


def plot_correlation_matrix2(data):
    numeric_data = data.select_dtypes(include=[np.number])
    cols = numeric_data.columns
    n = len(cols)

    # Initialize correlation and p-value matrices
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    # Compute correlation coefficients and p-values
    for i in range(n):
        for j in range(n):
            x = numeric_data.iloc[:, i]
            y = numeric_data.iloc[:, j]
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                valid = x.notna() & y.notna()
                if valid.sum() > 1:
                    corr, pval = pearsonr(x[valid], y[valid])
                    corr_matrix[i, j] = corr
                    pval_matrix[i, j] = pval
                else:
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan

    # Create annotation strings
    annot_matrix = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if np.isnan(val):
                annot_matrix[i, j] = ""
            else:
                star = "*" if pval_matrix[i, j] >= 0.05 and i != j else ""
                annot_matrix[i, j] = f"{val:.2f}{star}"

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=annot_matrix, fmt="", cmap="coolwarm",
                cbar=True, xticklabels=cols, yticklabels=cols, annot_kws={'size': 11})
    # plt.title('Correlation Matrix (p < 0.05 marked with *)', fontsize=14)
    plt.title('Correlation Matrix (p ≥ 0.05 marked with *)', fontsize=14)
    plt.xticks(fontsize=11, rotation=45)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_extreme_days(df, date_column, target_variable, num_days=3):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    daily_max = df[target_variable].resample('D').max()
    highest_days = daily_max.nlargest(num_days).index
    lowest_days = daily_max.nsmallest(num_days).index

    highest_days_df = df[df.index.normalize().isin(highest_days)]
    lowest_days_df = df[df.index.normalize().isin(lowest_days)]

    for day in highest_days:
        day_df = highest_days_df[highest_days_df.index.normalize() == day]
        ax = day_df.plot(subplots=True, figsize=(
            10, 12), title=f'Variables for {day.date()} (Highest Concentration)')
        for a in ax:
            a.title.set_fontsize(13)
            a.xaxis.label.set_size(12)
            a.yaxis.label.set_size(12)
        plt.tight_layout()
        plt.show()

    for day in lowest_days:
        day_df = lowest_days_df[lowest_days_df.index.normalize() == day]
        ax = day_df.plot(subplots=True, figsize=(
            10, 12), title=f'Variables for {day.date()} (Lowest Concentration)')
        for a in ax:
            a.title.set_fontsize(13)
            a.xaxis.label.set_size(12)
            a.yaxis.label.set_size(12)
        plt.tight_layout()
        plt.show()


# def plot_histograms(data, plots_per_row=4):
#     numeric_data = data.select_dtypes(
#         include=[np.number])  # Select only numeric columns
#     num_columns = len(numeric_data.columns)

#     # Calculate number of rows needed
#     num_rows = (num_columns + plots_per_row - 1) // plots_per_row
#     fig, axes = plt.subplots(num_rows, plots_per_row,
#                              figsize=(plots_per_row * 3, num_rows * 3))
#     axes = axes.flatten()

#     for i, column in enumerate(numeric_data.columns):
#         unique_values = numeric_data[column].nunique()
#         numeric_data[column].hist(ax=axes[i], bins=48, edgecolor='black')
#         axes[i].set_title(f"{column}\n(Unique: {unique_values})")
#         axes[i].set_xlabel('Value')
#         axes[i].set_ylabel('Frequency')

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()


# def plot_histograms_by_year(annual_dfs, plots_per_row=4):
#     """
#     Plots histograms of each variable, overlaying distributions per year.

#     Args:
#         annual_dfs (dict): Dictionary with keys as years and values as DataFrames.
#         plots_per_row (int): Number of plots per row in the figure.
#     """

#     df_all = (
#         pd.concat([df.assign(year=year) for year, df in annual_dfs.items()])
#         .reset_index(drop=True)
#     )

#     numeric_columns = df_all.select_dtypes(
#         include=[np.number]).columns.tolist()

#     if "year" in numeric_columns:
#         numeric_columns.remove("year")

#     num_columns = len(numeric_columns)
#     num_rows = (num_columns + plots_per_row - 1) // plots_per_row
#     fig, axes = plt.subplots(num_rows, plots_per_row,
#                              figsize=(plots_per_row * 4, num_rows * 3))
#     axes = axes.flatten()

#     for i, column in enumerate(numeric_columns):
#         sns.histplot(data=df_all, x=column, hue="year", bins=48,
#                      edgecolor="black", alpha=0.7, ax=axes[i])
#         axes[i].set_title(f"{column} Distribution per Year")
#         axes[i].set_xlabel("Value")
#         axes[i].set_ylabel("Frequency")

#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()


# def plot_correlation_matrix(data):
#     numeric_data = data.select_dtypes(
#         include=[np.number])  # Select only numeric columns
#     correlation_matrix = numeric_data.corr()
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(correlation_matrix, annot=True,
#                 fmt=".2f", cmap='coolwarm', cbar=True)
#     plt.title('Correlation Matrix')
#     plt.show()


# def plot_extreme_days(df, date_column, target_variable, num_days=3):
#     """
#     Plots the specified number of days with the highest and lowest peak values of the target variable.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame.
#     date_column (str): The name of the date column in the DataFrame.
#     target_variable (str): The name of the target variable column.
#     num_days (int): The number of days to plot for highest and lowest peak values (default is 3).

#     Returns:
#     None
#     """
#     df = df.copy()
#     df[date_column] = pd.to_datetime(df[date_column])

#     df.set_index(date_column, inplace=True)

#     daily_max = df[target_variable].resample('D').max()

#     highest_days = daily_max.nlargest(num_days).index
#     lowest_days = daily_max.nsmallest(num_days).index

#     highest_days_df = df[df.index.normalize().isin(highest_days)]
#     lowest_days_df = df[df.index.normalize().isin(lowest_days)]

#     for day in highest_days:
#         day_df = highest_days_df[highest_days_df.index.normalize() == day]
#         day_df.plot(subplots=True, figsize=(10, 12),
#                     title=f'Variables for {day.date()} (Highest Concentration)')
#         plt.tight_layout()
#         plt.show()

#     for day in lowest_days:
#         day_df = lowest_days_df[lowest_days_df.index.normalize() == day]
#         day_df.plot(subplots=True, figsize=(10, 12),
#                     title=f'Variables for {day.date()} (Lowest Concentration)')
#         plt.tight_layout()
#         plt.show()


def summarize_nan(df):
    """
    Summarizes NaN values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame summarizing the number and percentage of NaN values for each column.
    """
    nan_summary = df.isna().sum().reset_index()
    nan_summary.columns = ['Column', 'NaN Count']
    nan_summary['NaN Percentage'] = (nan_summary['NaN Count'] / len(df)) * 100

    nan_summary = nan_summary[nan_summary['NaN Count'] > 0]

    return nan_summary
