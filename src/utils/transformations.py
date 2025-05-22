import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


def h2o_concentration_vol(temp_C, RH, P_total=1013.25):
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


def transform_wind(df, method="xy"):
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


def add_time_components(df, date_column, start_date=None):
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

    return df


def add_time_features(df, date_column="Datetime"):
    """
    Add cyclic time features (sin/cos of time of day and period).
    """
    df = add_time_components(df, date_column)  # Assuming this exists already
    day_range = df["days_from_start"].max() - df["days_from_start"].min()

    df["cos_time"] = np.cos(df["time_of_day"] * 2 * np.pi / 1440)
    df["sin_time"] = np.sin(df["time_of_day"] * 2 * np.pi / 1440)
    df["cos_period"] = np.cos(df["days_from_start"] * 2 * np.pi / day_range)
    df["sin_period"] = np.sin(df["days_from_start"] * 2 * np.pi / day_range)

    df.drop(columns=["time_of_day", "days_from_start",
            "month", "day"], inplace=True, errors="ignore")
    return df


def add_h2o_concentration(df):
    """
    Add water vapor concentration feature.
    """
    df["h2o"] = h2o_concentration_vol(df["VAR_META.TDRY0"], df["VAR_META.RH0"])
    return df


def transform_features(df, log_features=None, minmax_features=None, standard_features=None,  quantile_features=None):
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
