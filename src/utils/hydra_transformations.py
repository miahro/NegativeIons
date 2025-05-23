import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


def h2o_concentration_vol_tetens(temp_C, RH, P_total=1013.25):
    """
    Convert temperature (°C) and relative humidity (%) to water vapor concentration (vol-%).

    Parameters:
        temp_C : float or array-like
            Temperature in degrees Celsius
        RH : float or array-like
            Relative humidity in %
        P_total : float, optional
            Total atmospheric pressure in hPa (default: 1013.25 hPa)

    Returns:
        H2O_vol_percent : float
            Water vapor concentration in volume %
    """
    # Calculate saturation vapor pressure using Tetens formula
    P_sat = 6.1078 * 10**((7.5 * temp_C) / (temp_C + 237.3))  # hPa

    # Actual water vapor pressure
    P_H2O = (RH / 100) * P_sat  # hPa

    # Water vapor concentration in volume percent
    H2O_vol_percent = (P_H2O / P_total) * 100  # vol-%

    return H2O_vol_percent


def h2o_concentration_vol_goff_gratch(temp_C, RH, P_total=1013.25):
    """
    Convert temperature (°C) and relative humidity (%) to water vapor concentration (vol-%)
    using the more accurate Goff-Gratch formulation.

    Parameters:
        temp_C : float or array-like
            Temperature in degrees Celsius
        RH : float or array-like
            Relative humidity in %
        P_total : float, optional
            Total atmospheric pressure in hPa (default: 1013.25 hPa)

    Returns:
        H2O_vol_percent : float
            Water vapor concentration in volume %
    """
    print("Calculating water vapor concentration using Goff-Gratch equation")

    temp_K = temp_C + 273.15  # Convert to Kelvin

    # Goff-Gratch equation for saturation vapor pressure over liquid water
    P_sat = 10 ** (
        -7.90298 * (373.16 / temp_K - 1) +
        5.02808 * np.log10(373.16 / temp_K) -
        1.3816e-7 * (10 ** (11.344 * (1 - temp_K / 373.16)) - 1) +
        8.1328e-3 * (10 ** (-3.49149 * (373.16 / temp_K - 1)) - 1) +
        np.log10(1013.25)  # Reference pressure in hPa
    )

    # Convert to actual water vapor pressure
    P_H2O = (RH / 100) * P_sat  # hPa

    # Calculate water vapor concentration in volume %
    H2O_vol_percent = (P_H2O / P_total) * 100  # vol-%

    return H2O_vol_percent


def transform_wind_hydra(df, method="xy"):
    """
    Transform wind speed and direction.
    Methods:
    - "xy"  -> Converts to x and y components (default)
    - "cos_sin" -> Converts wind direction to cos and sin, keeps wind speed
    - None  -> Keeps wind variables unchanged
    """
    df = df.copy()  # Avoid modifying original DataFrame

    if method == "xy":
        df["windX"] = df["VAR_META.WS00"] * \
            np.cos(np.radians(df["VAR_META.WDIR"]))
        df["windY"] = df["VAR_META.WS00"] * \
            np.sin(np.radians(df["VAR_META.WDIR"]))
        df.drop(columns=["VAR_META.WS00", "VAR_META.WDIR"], inplace=True)

    elif method == "cos_sin":
        df["wind_cos"] = np.cos(np.radians(df["VAR_META.WDIR"]))
        df["wind_sin"] = np.sin(np.radians(df["VAR_META.WDIR"]))
        df.drop(columns=["VAR_META.WDIR"], inplace=True)
    elif method == "xy_w_speed":
        df["windX"] = df["VAR_META.WS00"] * \
            np.cos(np.radians(df["VAR_META.WDIR"]))
        df["windY"] = df["VAR_META.WS00"] * \
            np.sin(np.radians(df["VAR_META.WDIR"]))
        df.drop(columns=["VAR_META.WDIR"], inplace=True)

    return df


def add_time_components_hydra(df, date_column, start_date=None):
    """
    Adds month, time of day, and days from start date components to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a date column.
    date_column (str): The name of the date column in the DataFrame.
    start_date (str or None): The start date in 'MM-DD' format. If None, the first day of the first month in the data is used.

    Returns:
    pd.DataFrame: The DataFrame with added month, time of day, and days from start date columns.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day

    df['time_of_day'] = df[date_column].dt.hour * \
        60 + df[date_column].dt.minute

    # Determine the start date
    if start_date is None:
        start_date = df[date_column].min().strftime('%m-%d')
    # Use a common year to ignore the year component
    start_date = pd.to_datetime(f'2000-{start_date}')

    # Calculate the number of days from the start date
    df['days_from_start'] = df.apply(lambda row: (pd.to_datetime(
        f'2000-{row["month"]:02d}-{row["day"]:02d}') - start_date).days, axis=1)
    df['days_from_start'] = df['days_from_start'].astype('float64')
    return df


def add_time_features_hydra(df, date_column="Datetime", period_method="cyclic"):
    """
    Add cyclic time features (sin/cos of time of day and period).
    """
    df = add_time_components_hydra(
        df, date_column)  # Assuming this exists already
    day_range = df["days_from_start"].max() - df["days_from_start"].min()

    df["cos_time"] = np.cos(df["time_of_day"] * 2 * np.pi / 1440)
    df["sin_time"] = np.sin(df["time_of_day"] * 2 * np.pi / 1440)

    if period_method == "cyclic":
        print("Adding cyclic period features")
        df["cos_period"] = np.cos(
            df["days_from_start"] * 2 * np.pi / day_range)
        df["sin_period"] = np.sin(
            df["days_from_start"] * 2 * np.pi / day_range)
        df.drop(columns=["days_from_start"], inplace=True, errors="ignore")
    elif period_method == "linear":
        print("Adding linear period features")
        df.drop(columns=["time_of_day", "month", "day"],
                inplace=True, errors="ignore")
    elif period_method == "both":
        print("Adding both cyclic and linear period features")
        df["cos_period"] = np.cos(
            df["days_from_start"] * 2 * np.pi / day_range)
        df["sin_period"] = np.sin(
            df["days_from_start"] * 2 * np.pi / day_range)
    elif period_method == "hybrid":
        print("Adding both cyclic (only cosine) and linear period features")
        df["cos_period"] = np.cos(
            df["days_from_start"] * 2 * np.pi / day_range)
    elif period_method == "cos":
        print("Adding only cosine period features")
        df["cos_period"] = np.cos(
            df["days_from_start"] * 2 * np.pi / day_range)
        df.drop(columns=["days_from_start"],
                inplace=True, errors="ignore")
    elif period_method == "sin":
        print("Adding only sin period features")
        df["sin_period"] = np.sin(
            df["days_from_start"] * 2 * np.pi / day_range)
        df.drop(columns=["days_from_start"],
                inplace=True, errors="ignore")

    df.drop(columns=["time_of_day", "month", "day"],
            inplace=True, errors="ignore")

    return df


def add_h2o_concentration_hydra(df, method="tetens"):
    """
    Add water vapor concentration feature.
    """
    print(f"Adding water vapor concentration feature using {method} method")
    if method == "tetens":
        df["h2o"] = h2o_concentration_vol_tetens(
            df["VAR_META.TDRY0"], df["VAR_META.RH0"])
    elif method == "goff_gratch":
        df["h2o"] = h2o_concentration_vol_goff_gratch(
            df["VAR_META.TDRY0"], df["VAR_META.RH0"])
    return df


def transform_features_hydra(df, log_features=None, minmax_features=None, standard_features=None,  quantile_features=None):
    """
    Apply different transformations to features and target.

    Parameters:
    - df: DataFrame containing features and target.
    - target_col: Name of the target column.
    - scale_method: "standard" (StandardScaler) or "quantile" (Gaussian QuantileTransformer).
    - log_features: List of features to apply log(1+x) transformation.
    - minmax_features: List of features to apply MinMaxScaler.
    - date_column: Name of the date column (if present) to be ignored in transformations.

    Returns:
    - df_transformed: Transformed DataFrame.
    - scaler: Fitted scaler for future use.
    """

    df_transformed = df.copy()

    if log_features:
        for col in log_features:
            if col in df_transformed.columns:
                df_transformed[col] = np.log1p(df_transformed[col])

    scaler = None

    if standard_features:
        scaler = StandardScaler()
        for col in standard_features:
            if col in df_transformed.columns:
                df_transformed[col] = scaler.fit_transform(
                    df_transformed[[col]])

    if quantile_features:
        scaler = QuantileTransformer(output_distribution='normal')
        for col in quantile_features:
            if col in df_transformed.columns:
                df_transformed[col] = scaler.fit_transform(
                    df_transformed[[col]])

    if minmax_features:
        minmax_scaler = MinMaxScaler()
        for col in minmax_features:
            if col in df_transformed.columns:
                df_transformed[col] = minmax_scaler.fit_transform(
                    df_transformed[[col]])

    return df_transformed, scaler


def filter_out_rain(df, rain_col, threshold=0.01):
    """
    Filter out rows with rain below a certain threshold.
    """
    df = df.copy()
    df = df[df[rain_col] < threshold]
    df.drop(columns=[rain_col], inplace=True)
    return df


def add_time_lag_feat(df, feat_col, lags=[]):
    """
    Add time lag features to the DataFrame.
    """
    df = df.copy()
    for lag in lags:
        df[f"{feat_col}_lag{lag*30}"] = df[feat_col].shift(lag)
    return df


def filter_date_range(df, timestamp_col, start_date, end_date):
    """
    Filters a dataframe to include only data within the specified date range for each year.

    Args:
        df (pd.DataFrame): The input dataframe containing a timestamp column.
        timestamp_col (str): The name of the timestamp column.
        start_date (str): Start date (format: 'MM-DD', e.g., '06-01' for June 1st).
        end_date (str): End date (format: 'MM-DD', e.g., '08-31' for August 31st).

    Returns:
        pd.DataFrame: Filtered dataframe containing only the specified date range for each year.
    """

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["month_day"] = df[timestamp_col].dt.strftime("%m-%d")
    filtered_df = df[(df["month_day"] >= start_date) & (
        df["month_day"] <= end_date)].drop(columns=["month_day"])

    return filtered_df


def drop_initial_lagged_rows(df, timestamp_col, max_lag):
    """
    Drops the first `max_lag` rows for each year in the dataset to prevent time leakage from previous years.

    Args:
        df (pd.DataFrame): The input dataframe.
        timestamp_col (str): The name of the timestamp column.
        max_lag (int): The maximum lag applied in time-lagged features.

    Returns:
        pd.DataFrame: The dataframe with initial lagged rows removed per year.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col])  # Ensure datetime format
    df['Year'] = df[timestamp_col].dt.year  # Extract year

    # Group by year and remove first max_lag rows for each year
    df = df.groupby('Year').apply(
        lambda x: x.iloc[max_lag:]).reset_index(drop=True)

    # Drop the 'Year' column as it's no longer needed
    df.drop(columns=['Year'], inplace=True)

    return df


def convert_feature_names(feature_dict, to_yaml=True):
    """
    Convert feature names between YAML-safe and original format.

    Args:
        feature_dict (dict): Dictionary of features with lags.
        to_yaml (bool): If True, converts from "." to "__" (for YAML compatibility).
                        If False, converts from "__" back to "." (for MLflow & model input).

    Returns:
        dict: Dictionary with converted feature names.
    """
    if to_yaml:
        return {feat.replace(".", "__"): lags for feat, lags in feature_dict.items()}
    else:
        return {feat.replace("__", "."): lags for feat, lags in feature_dict.items()}


def interpolate_features(df, linear_features, circular_features=None, method="linear"):
    """
    Imputes missing values in the dataset using the specified method.

    Args:
        df (pd.DataFrame): Dataframe containing raw features.
        linear_features (list): List of linear features to interpolate.
        circular_features (list, optional): List of circular features (e.g., wind direction) to interpolate.
        method (str): Interpolation method. Default is "linear".

    Returns:
        pd.DataFrame: Imputed dataframe.
    """

    print("Debug: interpolate_features running")
    print(f"Linear features: {linear_features}")
    print(f"Circular features: {circular_features}")
    print(f"Method: {method}")

    df = df.copy()

    # Standard linear interpolation for most features
    for feature in linear_features:
        if feature in df.columns:
            df[feature] = df[feature].interpolate(
                method=method, limit_direction="both")

    # Special handling for wind direction (circular interpolation)
    if circular_features is not None:
        for feature in circular_features:
            if feature in df.columns:
                # Convert angles to unit vectors
                x = np.cos(np.radians(df[feature]))
                y = np.sin(np.radians(df[feature]))

                # Interpolate x and y components separately
                x_interp = pd.Series(x).interpolate(
                    method=method, limit_direction="both")
                y_interp = pd.Series(y).interpolate(
                    method=method, limit_direction="both")

                # Normalize interpolated values
                valid_mask = ~x_interp.isna() & ~y_interp.isna()
                # Initialize output array
                angle_interp = np.full_like(df[feature], np.nan)
                angle_interp[valid_mask] = np.degrees(
                    np.arctan2(y_interp[valid_mask], x_interp[valid_mask]))

                # Ensure values remain in the range [0, 360]
                df[feature] = (angle_interp + 360) % 360

    return df
