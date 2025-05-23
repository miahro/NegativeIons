"""Module for utility functions for Hydra."""

import pandas as pd
import numpy as np
import os
import hydra
from omegaconf import OmegaConf

# from utils.transformations import transform_wind, add_time_features, add_h2o_concentration
from utils.hydra_modeling import split_train_test_hydra, split_train_test_years_hydra, split_train_test_days_hydra
from utils.hydra_transformations import transform_features_hydra, transform_wind_hydra, add_time_features_hydra, add_h2o_concentration_hydra, filter_out_rain, add_time_lag_feat, filter_date_range, drop_initial_lagged_rows, convert_feature_names, interpolate_features
# from utils.modeling import split_train_test


def prepare_data_hydra(cfg):
    """
    Loads data, applies selected transformations, and splits into train/test.
    Args:
        cfg (OmegaConf): The Hydra config object.
    Returns:
        pd.DataFrame: The prepared DataFrame.
    """
    import os
    import pandas as pd

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(
        SCRIPT_DIR, "../.."))
    cfg.data.file_path = os.path.join(PROJECT_ROOT, cfg.data.file_path)

    df = pd.read_csv(cfg.data.file_path)

    df = filter_date_range(df, cfg.data.timestamp,
                           cfg.data.start_date, cfg.data.end_date)

    if cfg.data.year:
        df = df[df[cfg.data.timestamp].dt.year == cfg.data.year]
    elif cfg.data.years:
        df = df[df[cfg.data.timestamp].dt.year.isin(cfg.data.years)]

    if cfg.preprocessing.transformations.add_year:
        df['year'] = df[cfg.data.timestamp].dt.year
        df = pd.get_dummies(df, columns=['year'], prefix='year')

    for col in cfg.data.drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print(
        f"preprocessing.missing_value_strategy.selected: {cfg.preprocessing.missing_value_strategy.selected}")

    if cfg.preprocessing.missing_value_strategy.selected == "interpolation":
        interpolation_cfg = cfg.preprocessing.missing_value_strategy.options.interpolation
        df = interpolate_features(df,
                                  linear_features=interpolation_cfg.linear_features,
                                  circular_features=interpolation_cfg.circular_features,
                                  method=interpolation_cfg.method)

    if cfg.preprocessing.transformations.filter_rain:
        df = filter_out_rain(df, cfg.preprocessing.transformations.rain_column,
                             cfg.preprocessing.transformations.rain_threshold)

    if cfg.preprocessing.transformations.filter_par:
        df = df[df[cfg.preprocessing.transformations.par_column]
                > cfg.preprocessing.transformations.par_threshold]

    wind_method = cfg.preprocessing.transformations.wind_method

    if wind_method:
        df = transform_wind_hydra(df, method=wind_method)

    if cfg.preprocessing.transformations.add_time:
        period_method = cfg.preprocessing.transformations.period_method
        df = add_time_features_hydra(df, period_method=period_method)
    # elif cfg.data.timestamp:
    #     df.drop(columns=[cfg.data.timestamp], inplace=True)
    if cfg.preprocessing.transformations.add_h2o:
        h2o_method = cfg.preprocessing.transformations.h2o_method
        df = add_h2o_concentration_hydra(df, method=h2o_method)

    if cfg.preprocessing.transformations.add_time_lags:
        max_lag = 0

        log_features = cfg.preprocessing.normalizations.log_features or []
        minmax_features = cfg.preprocessing.normalizations.minmax_features or []
        standard_features = cfg.preprocessing.normalizations.standard_features or []
        quantile_features = cfg.preprocessing.normalizations.quantile_features or []

        for feat, lags in cfg.preprocessing.transformations.time_lagged_features.items():
            feat = feat.replace("__", ".")
            print(f"Feature: {feat}, Lags: {lags}")

            if lags:
                df = add_time_lag_feat(df, feat, lags)
                if max(lags) > max_lag:
                    max_lag = max(lags)

                for lag in lags:
                    lagged_feat = f"{feat}_lag{lag*30}"  # Time lag in minutes

                    if feat in log_features:
                        log_features.append(lagged_feat)
                    if feat in minmax_features:
                        minmax_features.append(lagged_feat)
                    if feat in standard_features:
                        standard_features.append(lagged_feat)
                    if feat in quantile_features:
                        quantile_features.append(lagged_feat)

        print(f"Max Lag: {max_lag}")

        df = drop_initial_lagged_rows(df, cfg.data.timestamp, max_lag)

        df.dropna(inplace=True)

        cfg.preprocessing.normalizations.log_features = list(set(log_features))
        cfg.preprocessing.normalizations.minmax_features = list(
            set(minmax_features))
        cfg.preprocessing.normalizations.standard_features = list(
            set(standard_features))
        cfg.preprocessing.normalizations.quantile_features = list(
            set(quantile_features))

    df.dropna(inplace=True)
    df.rename(columns=lambda x: x.replace("__", "."), inplace=True)

    if cfg.split.year_split:
        train_df, test_df = split_train_test_years_hydra(
            df, timestamp_col=cfg.data.timestamp, test_years=cfg.split.test_years)
    elif cfg.split.day_split:
        train_df, test_df = split_train_test_days_hydra(
            df, timestamp_col=cfg.data.timestamp)
    else:
        train_df, test_df = split_train_test_hydra(
            df, cfg.split.test_size, cfg.split.random_state, cfg.split.shuffle)

    if cfg.data.permute_target_random or cfg.data.permute_target_day:
        print("Permuting target values")
        train_df[cfg.data.target] = permute_target(
            train_df[cfg.data.target],
            timestamp=train_df[cfg.data.timestamp],
            permute_random=cfg.data.permute_target_random,
            permute_day=cfg.data.permute_target_day
        )

    train_df_scaled, _ = transform_features_hydra(
        df=train_df,
        log_features=cfg.preprocessing.normalizations.log_features,
        minmax_features=cfg.preprocessing.normalizations.minmax_features,
        standard_features=cfg.preprocessing.normalizations.standard_features,
        quantile_features=cfg.preprocessing.normalizations.quantile_features
    )

    test_df_scaled, _ = transform_features_hydra(
        df=test_df,
        log_features=cfg.preprocessing.normalizations.log_features,
        minmax_features=cfg.preprocessing.normalizations.minmax_features,
        standard_features=cfg.preprocessing.normalizations.standard_features,
        quantile_features=cfg.preprocessing.normalizations.quantile_features
    )

    X_train = train_df_scaled.drop(columns=[cfg.data.target])
    y_train = train_df_scaled[cfg.data.target]
    X_test = test_df_scaled.drop(columns=[cfg.data.target])
    y_test = test_df_scaled[cfg.data.target]

    print(f"columns returned by prepare_data_hydra: {X_train.columns}")

    return X_train, y_train, X_test,  y_test


def permute_target(y, timestamp=None, permute_random=False, permute_day=False):
    """
    Permute target values while preserving index alignment.

    Args:
        y (pd.Series): Target values.
        timestamp (pd.Series): Timestamp column for day-based permutation.
        permute_random (bool): Whether to fully randomize target values.
        permute_day (bool): Whether to shuffle within day groups.

    Returns:
        pd.Series: Permuted target values with preserved index.
    """
    if permute_random:
        print("Permuting target: FULL RANDOM")
        # Extract permuted values
        permuted_values = y.sample(frac=1, random_state=42).values
        # Rebuild with original index
        y = pd.Series(permuted_values, index=y.index)

    elif permute_day and timestamp is not None:
        print("Permuting target: WITHIN DAY")
        y = y.groupby(timestamp.dt.date).transform(
            lambda x: pd.Series(np.random.permutation(x.values), index=x.index)
        )

    return y
