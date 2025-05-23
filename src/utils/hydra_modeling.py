# Make sure evaluate_model is correctly imported
# from utils.modeling import evaluate_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from omegaconf import OmegaConf
from sklearn.model_selection import TimeSeriesSplit, GroupKFold


def split_train_test_hydra(df, test_size=0.2, random_state=42, shuffle=True):
    """Splits a DataFrame into training and testing sets while keeping target variable inside."""
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=shuffle)

    return df_train, df_test


def split_train_test_years_hydra(df, timestamp_col, test_years=[2024]):
    """Splits a DataFrame into training and testing sets based on years."""

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["year"] = df[timestamp_col].dt.year

    test_years = set(test_years)
    df_train = df[~df["year"].isin(test_years)]
    df_test = df[df["year"].isin(test_years)]

    print(
        f"in split df_train.shape: {df_train.shape}, train_years: {df_train['year'].unique()}")
    print(
        f"in split df_test.shape: {df_test.shape}, test_years: {df_test['year'].unique()}")

    return df_train.drop(columns=["year"]), df_test.drop(columns=["year"])


def split_train_test_days_hydra(df, timestamp_col, n_splits=5):
    """Splits a DataFrame into training and testing sets based on full days."""

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df["day_int"] = pd.factorize(df[timestamp_col].dt.strftime('%Y-%m-%d'))[0]

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(df, groups=df['day_int']):
        train_set = df.iloc[train_idx]
        test_set = df.iloc[test_idx]
        break
    train_set.drop(columns=["day_int"], inplace=True)
    test_set.drop(columns=["day_int"], inplace=True)

    return train_set, test_set


def log_mlflow_experiment_params(cfg, model_name, model_params):
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def restore_feature_names(flattened_cfg):
        restored_cfg = {}
        for key, value in flattened_cfg.items():
            restored_key = key.replace("__", ".")
            restored_cfg[restored_key] = value
        return restored_cfg

    # Convert OmegaConf to a standard dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flattened_cfg = flatten_dict(cfg_dict)
    cleaned_cfg = restore_feature_names(flattened_cfg)

    def get_short_key(full_key):
        parts = full_key.split('.')
        if len(parts) > 1:
            # Truncate parts[-2] to max 4 characters + keep full last part
            short_part = parts[-2][:4]
            return f"{short_part}.{parts[-1]}"
        return full_key

    # Collect shortened keys into a dictionary
    short_params = {}
    for key, value in cleaned_cfg.items():
        short_key = get_short_key(key)

        # Ensure keys are unique and within MLflow limit
        if short_key in short_params:
            print(f"⚠️ Skipping duplicate key: {short_key}")  # Debug info
            continue
        if len(short_key) > 250:
            print(f"⚠️ Skipping too-long key: {short_key}")  # Debug info
            continue

        if isinstance(value, list):
            value = ", ".join(map(str, value)) if value else "None"
        if isinstance(value, (int, float, str, bool, type(None))):
            short_params[short_key] = value

    if short_params:
        mlflow.log_params(short_params)

    if model_params:
        mlflow.log_params(model_params)

    mlflow.log_param("model_type", model_name)


def log_mlflow_results(model_name, metrics, best_params=None, cv_r2=None):
    """
    Logs model evaluation results to MLflow.

    Args:
        model_name (str): Name of the model.
        metrics (dict): Dictionary containing evaluation metrics.
        best_params (dict, optional): Best hyperparameters if grid search was used.
        cv_r2 (float, optional): Cross-validation R² score.
    """
    mlflow.log_metric("r2_train", metrics["r2_train"])
    mlflow.log_metric("r2_val", metrics["r2_val"])
    mlflow.log_metric("rmse_val", metrics["rmse_val"])
    mlflow.log_metric("mae_val", metrics["mae_val"])

    if cv_r2 is not None:
        mlflow.log_metric("cv_r2", cv_r2)

    if best_params:
        for param, value in best_params.items():
            mlflow.set_tag(f"best_{param}", value)


def evaluate_model_hydra(model, X_train, y_train, X_val, y_val):
    """Evaluate model performance and return metrics."""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_val": r2_score(y_val, y_val_pred),
        "rmse_val": root_mean_squared_error(y_val, y_val_pred),
        "mae_val": mean_absolute_error(y_val, y_val_pred)
    }

    return metrics


def train_and_evaluate_model(cfg, model_cfg, model, X_train, y_train, X_val, y_val):
    """
    Trains the model, performs optional hyperparameter tuning, and evaluates performance.

    Args:
        cfg (OmegaConf): Hydra config.
        model_cfg (dict): The model's config.
        model: The initialized machine learning model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.

    Returns:
        model: Trained model (best estimator if search was used).
        best_params (dict or None): Best parameters if hyperparameter tuning was performed.
        cv_r2 (float or None): Cross-validation R² score if CV was used.
        metrics (dict): Dictionary of model evaluation metrics.
    """
    best_params = None
    cv_r2 = None

    if "grid_search" in model_cfg and model_cfg.grid_search.enabled:
        print(
            f"🔍 Performing {cfg.training.search_method.capitalize()} Search for {model_cfg.name}...")

        param_grid = OmegaConf.to_container(
            model_cfg.grid_search.param_grid, resolve=True)

        if cfg.training.search_method == "grid":
            search = GridSearchCV(
                model, param_grid, scoring=cfg.training.scoring, cv=cfg.training.cv_folds, n_jobs=-1, return_train_score=True
            )
        elif cfg.training.search_method == "random":
            search = RandomizedSearchCV(
                model, param_grid, n_iter=cfg.training.n_iter, scoring=cfg.training.scoring, cv=cfg.training.cv_folds, n_jobs=-1, random_state=42, return_train_score=True
            )
        else:
            raise ValueError(
                "❌ Invalid search_method. Choose 'grid' or 'random'.")

        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_

        cv_r2 = search.cv_results_['mean_test_score'].max()

    else:
        model.fit(X_train, y_train)
        if cfg.training.use_cv:
            if cfg.training.use_time_series_cv:
                tscv = TimeSeriesSplit(n_splits=cfg.training.cv_folds)
                cv_r2 = np.mean(cross_val_score(
                    model, X_train, y_train, cv=tscv, scoring=cfg.training.scoring))

            else:
                cv_r2 = np.mean(cross_val_score(
                    model, X_train, y_train, cv=cfg.training.cv_folds, scoring=cfg.training.scoring))

    metrics = evaluate_model_hydra(model, X_train, y_train, X_val, y_val)

    return model, best_params, cv_r2, metrics


def perform_regression_mlflow_hydra(cfg, X_train, y_train, X_val, y_val, printout=True):
    """
    Perform regression using models from Hydra config, track results in MLflow.

    Args:
        cfg (OmegaConf): Hydra config containing model details.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
    """

    print(f"X_train columns: {X_train.columns}")

    X_train = X_train.select_dtypes(exclude=["datetime64"])
    X_val = X_val.select_dtypes(exclude=["datetime64"])

    results_list = []
    degree = cfg.preprocessing.polynomial_features.degree if "polynomial_features" in cfg.preprocessing else None
    if degree:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

    for model_cfg in cfg.models:
        model_name = model_cfg.name
        model_params = model_cfg.params or {}

        print(f"model name: {model_name}")
        print(f"model params: {model_params}")
        print(
            f"🛠️ Grid Search Enabled: {model_cfg.get('grid_search', {}).get('enabled', False)}")
        print(
            f"📊 Grid Search Param Grid: {model_cfg.get('grid_search', {}).get('param_grid', {})}")

        model = eval(model_name)(**model_params)

        with mlflow.start_run():
            log_mlflow_experiment_params(cfg, model_name, model_params)

            # Train and evaluate model
            model, best_params, cv_r2, metrics = train_and_evaluate_model(
                cfg, model_cfg, model, X_train, y_train, X_val, y_val
            )
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_val))

            # Log results
            print(
                f"period_method: {cfg.preprocessing.transformations.period_method}")

            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
            elif hasattr(model, "coef_"):  # For linear models
                feature_importance = np.abs(
                    model.coef_)
            else:
                feature_importance = None

            # Convert to probability distribution if feature importance exists
            if feature_importance is not None:
                feature_importance = feature_importance / feature_importance.sum()  # Normalize

                feature_importance_df = pd.DataFrame({
                    "Feature": X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(len(feature_importance))],
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)

                # Save and log feature importance
                fi_csv_path = "feature_importance.csv"
                feature_importance_df.to_csv(fi_csv_path, index=False)
                mlflow.log_artifact(fi_csv_path)  # Saves the CSV in MLflow UI

                # Log feature importance as dictionary
                feature_importance_dict = feature_importance_df.set_index("Feature")[
                    "Importance"].to_dict()
                mlflow.log_dict(feature_importance_dict,
                                "feature_importance.json")

            log_mlflow_results(model_name, metrics, best_params, cv_r2)
            input_example = pd.DataFrame(X_train).iloc[0:1]
            mlflow.sklearn.log_model(
                model, model_name, input_example=input_example)

            results = {
                "model": model_name,
                "degree": degree if degree else "None",
                "best_params": best_params,
                "cv_r2": cv_r2,
                "r2_train": metrics["r2_train"],
                "r2_val": metrics["r2_val"],
                "rmse_val": metrics["rmse_val"],
                "mae_val": metrics["mae_val"]
            }
            results_list.append(results)

            if printout:
                print(f"\n🚀 Model: {model_name}")
                print(
                    f"Train R²: {metrics['r2_train']:.4f} | Validation R²: {metrics['r2_val']:.4f}")
                print(
                    f"Validation RMSE: {metrics['rmse_val']:.4f} | Validation MAE: {metrics['mae_val']:.4f}")
                if cv_r2 is not None:
                    print(f"Cross-Validation R²: {cv_r2:.4f}")
                if best_params:
                    print(f"Best Parameters: {best_params}")

    return pd.DataFrame(results_list)
