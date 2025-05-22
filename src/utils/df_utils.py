import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.transformations import transform_wind, add_time_features, add_h2o_concentration
from utils.modeling import split_train_test


def get_valid_date_stamps(df, date_column):
    """
    Returns a DataFrame with the first and last valid date stamps for each data column.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column and several data columns.
    date_column (str): The name of the date column in the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with columns 'data_column', 'first_valid_date', and 'last_valid_date'.
    """
    df[date_column] = pd.to_datetime(df[date_column])

    results = []

    for column in df.columns:
        if column != date_column:
            first_valid_index = df[column].first_valid_index()
            last_valid_index = df[column].last_valid_index()

            first_valid_date = df.loc[first_valid_index,
                                      date_column] if first_valid_index is not None else None
            last_valid_date = df.loc[last_valid_index,
                                     date_column] if last_valid_index is not None else None

            results.append({
                'data_column': column,
                'first_valid_date': first_valid_date,
                'last_valid_date': last_valid_date
            })

    results_df = pd.DataFrame(results)

    return results_df


def check_timestamp_continuity(df, date_column, interval='30min'):
    """
    Checks for continuity of timestamps in the date column and returns a list of missing timestamps.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column.
    date_column (str): The name of the date column in the DataFrame.
    interval (str): The expected interval between timestamps (default is '30min').

    Returns:
    list: A list of missing timestamps.
    """
    df[date_column] = pd.to_datetime(df[date_column])

    df = df.sort_values(by=date_column)

    full_date_range = pd.date_range(
        start=df[date_column].min(), end=df[date_column].max(), freq=interval)

    missing_timestamps = full_date_range.difference(df[date_column])

    return missing_timestamps.tolist()


def identify_missing_periods(df, date_column, interval='30min'):
    """
    Identifies missing periods for each variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column and several data columns.
    date_column (str): The name of the date column in the DataFrame.
    interval (str): The expected interval between timestamps (default is '30min').

    Returns:
    pd.DataFrame: A DataFrame summarizing the missing periods for each variable.
    """

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    full_date_range = pd.date_range(
        start=df[date_column].min(), end=df[date_column].max(), freq=interval)

    missing_periods = []

    for column in df.columns:
        if column != date_column:
            missing_timestamps = full_date_range.difference(
                df[date_column][df[column].notna()])
            if not missing_timestamps.empty:
                missing_periods.append({
                    'variable': column,
                    'missing_periods': missing_timestamps
                })

    missing_periods_df = pd.DataFrame(missing_periods)

    return missing_periods_df


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

def plot_missing_data_heatmap(df, date_column, transpose=True):
    """
    Plots a heatmap of missing data for each variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column and several data columns.
    date_column (str): The name of the date column in the DataFrame.
    """
    # Update default font size globally
    plt.rcParams.update({'font.size': 12})

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    missing_data = df.isna()

    plt.figure(figsize=(15, 10))

    if transpose:
        missing_data = missing_data.T
        ax = sns.heatmap(missing_data, cbar=False, cmap='viridis')
        plt.title('Transposed Missing Data Heatmap', fontsize=16)
        plt.xlabel('Timestamps', fontsize=14)
        plt.ylabel('Variables', fontsize=14)

        # Format x-tick labels to show every 3rd label
        xticks = ax.get_xticks()
        xticklabels = [pd.to_datetime(label.get_text()).strftime(
            '%Y-%m') for label in ax.get_xticklabels()]
        ax.set_xticks(xticks[::3])
        ax.set_xticklabels(xticklabels[::3], rotation=45, ha='right')
    else:
        ax = sns.heatmap(missing_data, cbar=True, cmap='viridis')
        plt.title('Missing Data Heatmap', fontsize=16)
        plt.xlabel('Variables', fontsize=14)
        plt.ylabel('Timestamps', fontsize=14)

        yticks = ax.get_yticks()
        yticklabels = [pd.to_datetime(label.get_text()).strftime(
            '%Y-%m') for label in ax.get_yticklabels()]
        ax.set_yticks(yticks[::3])
        ax.set_yticklabels(yticklabels[::3], fontsize=10)

    plt.tight_layout()
    plt.show()


# def plot_missing_data_heatmap(df, date_column, transpose=True):
#     """
#     Plots a heatmap of missing data for each variable in the DataFrame.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame with a date column and several data columns.
#     date_column (str): The name of the date column in the DataFrame.
#     """

#     df = df.copy()
#     df[date_column] = pd.to_datetime(df[date_column])
#     df.set_index(date_column, inplace=True)

#     missing_data = df.isna()

#     plt.figure(figsize=(15, 10))

#     if transpose:
#         missing_data = missing_data.T
#         ax = sns.heatmap(missing_data, cbar=False, cmap='viridis')
#         plt.title('Transposed Missing Data Heatmap')
#         plt.xlabel('Timestamps')
#         plt.ylabel('Variables')
#         ax.set_xticklabels([pd.to_datetime(label.get_text()).strftime(
#             '%Y-%m') for label in ax.get_xticklabels()])
#     else:
#         ax = sns.heatmap(missing_data, cbar=True, cmap='viridis')
#         plt.title('Missing Data Heatmap')
#         plt.xlabel('Variables')
#         plt.ylabel('Timestamps')
#         ax.set_yticklabels([pd.to_datetime(label.get_text()).strftime(
#             '%Y-%m') for label in ax.get_yticklabels()])

#     plt.show()


def filter_by_month_interval(df, date_column, start_month, end_month):
    """
    Filters the DataFrame to include only rows where the month is within the specified interval, regardless of the year.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column.
    date_column (str): The name of the date column in the DataFrame.
    start_month (int): The start month (1-12).
    end_month (int): The end month (1-12).

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    df['month'] = df[date_column].dt.month

    if start_month <= end_month:
        filtered_df = df[(df['month'] >= start_month)
                         & (df['month'] <= end_month)]
    else:
        filtered_df = df[(df['month'] >= start_month)
                         | (df['month'] <= end_month)]

    filtered_df = filtered_df.drop(columns=['month'])

    return filtered_df


def filter_by_year_interval(df, date_column, start_year, end_year):
    """
    Filters the DataFrame to include only rows where the year is within the specified interval.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column.
    date_column (str): The name of the date column in the DataFrame.
    start_year (int): The start year.
    end_year (int): The end year.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    filtered_df = df[(df[date_column].dt.year >= start_year)
                     & (df[date_column].dt.year <= end_year)]

    return filtered_df


def prepare_data(file_path, target_column, test_size=0.2, wind_method=None, add_time=False, add_h2o=False, dropna=True):
    """
    Loads data, applies selected transformations, and splits into train/test.

    Parameters:
    - file_path: Path to CSV file.
    - target_column: Name of the target column.
    - test_size: Test data ratio.
    - wind_method: Wind transformation method ("xy", "cos_sin", or None).
    - add_time: Whether to add time features.
    - add_h2o: Whether to add H2O concentration.

    Returns:
    - X_train, X_test, y_train, y_test
    """
    if dropna:
        df = pd.read_csv(file_path).dropna()
    else:
        df = pd.read_csv(file_path)

    if wind_method:
        df = transform_wind(df, method=wind_method)

    if add_time:
        df = add_time_features(df)

    if add_h2o:
        df = add_h2o_concentration(df)

    X_train, X_test, y_train, y_test = split_train_test(
        df, target_column, test_size)

    return X_train, X_test, y_train, y_test
