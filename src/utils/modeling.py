from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn


import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = np.mean(y)

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=np.float64)


def evaluate_model(model, X_train, y_train, X_val, y_val):
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


def perform_regression(model, X_train, y_train, X_val, y_val,
                       degree=None, date_column=None, param_grid=None,
                       cv=5, scoring='r2', printout=True, use_cv=False):
    """
    Perform regression with optional cross-validation and polynomial features.

    Parameters:
    - model: Regression model (any sklearn regressor).
    - X_train, y_train: Training data.
    - X_val, y_val: Validation data.
    - degree: Polynomial degree (if None, no polynomial features are used).
    - param_grid: Hyperparameter grid for tuning (if None, trains without GridSearchCV).
    - cv: Number of cross-validation folds.
    - scoring: Metric for GridSearchCV (default: R²).
    - printout: Whether to print results.
    - use_cv: If True, performs cross-validation on training data.

    Returns:
    - results: Dictionary with trained model, CV score (if used), and validation metrics.
    """

    if date_column:
        X_train = X_train.drop(columns=[date_column], errors='ignore')
        X_val = X_val.drop(columns=[date_column], errors='ignore')

    if degree:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

    if param_grid:
        grid_cv = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
        grid_cv.fit(X_train, y_train)
        model = grid_cv.best_estimator_
        best_params = grid_cv.best_params_
        cv_r2 = grid_cv.best_score_
    else:
        best_params = None
        if use_cv:
            cv_r2 = np.mean(cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring))
        else:
            cv_r2 = None

        model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

    row = {
        "model": model.__class__.__name__,
        "degree": degree if degree else "None",
        "best_params": best_params,
        "cv_r2": cv_r2,
        "r2_train": metrics["r2_train"],
        "r2_val": metrics["r2_val"],
        "rmse_val": metrics["rmse_val"],
        "mae_val": metrics["mae_val"]
    }

    if printout:
        print(f"Model: {model.__class__.__name__}")
        if best_params:
            print(f"Best Parameters: {best_params}")
            print(f"Cross-Validation R²: {cv_r2:.4f}")
        elif use_cv:
            print(f"Cross-Validation R²: {cv_r2:.4f}")
        print(f"Train R²: {metrics['r2_train']:.4f}")
        print(f"Validation R²: {metrics['r2_val']:.4f}")
        print(f"Validation RMSE: {metrics['rmse_val']:.4f}")
        print(f"Validation MAE: {metrics['mae_val']:.4f}")

    return row


def perform_regression_mlflow(model, X_train, y_train, X_val, y_val,
                              degree=None, date_column=None, param_grid=None,
                              cv=5, scoring='r2', printout=True, use_cv=False):
    """
    Perform regression with optional cross-validation and polynomial features.
    """

    # Remove date column if exists
    if date_column:
        X_train = X_train.drop(columns=[date_column], errors='ignore')
        X_val = X_val.drop(columns=[date_column], errors='ignore')

    # Apply polynomial features if specified
    if degree:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

    # Start MLflow experiment tracking
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("degree", degree if degree else "None")
        mlflow.log_param("use_cv", use_cv)
        mlflow.log_param("cv_folds", cv if use_cv else "None")
        mlflow.log_param("scoring_metric", scoring)

        if param_grid:
            grid_cv = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
            grid_cv.fit(X_train, y_train)
            model = grid_cv.best_estimator_
            best_params = grid_cv.best_params_
            cv_r2 = grid_cv.best_score_

            # Log best hyperparameters from GridSearchCV
            for param, value in best_params.items():
                mlflow.log_param(param, value)

            mlflow.log_metric("cv_r2", cv_r2)
        else:
            best_params = None
            if use_cv:
                cv_r2 = np.mean(cross_val_score(
                    model, X_train, y_train, cv=cv, scoring=scoring))
                mlflow.log_metric("cv_r2", cv_r2)
            else:
                cv_r2 = None

            model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log evaluation metrics
        mlflow.log_metric("r2_train", metrics["r2_train"])
        mlflow.log_metric("r2_val", metrics["r2_val"])
        mlflow.log_metric("rmse_val", metrics["rmse_val"])
        mlflow.log_metric("mae_val", metrics["mae_val"])

        # Save the trained model in MLflow
        # mlflow.sklearn.log_model(model, "model")
        # Create an example input (e.g., the first row of X_train)
        input_example = pd.DataFrame(
            X_train)  # Ensure it's a DataFrame
        input_example = input_example.iloc[0:1]
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        row = {
            "model": model.__class__.__name__,
            "degree": degree if degree else "None",
            "best_params": best_params,
            "cv_r2": cv_r2,
            "r2_train": metrics["r2_train"],
            "r2_val": metrics["r2_val"],
            "rmse_val": metrics["rmse_val"],
            "mae_val": metrics["mae_val"]
        }

        if printout:
            print(f"Model: {model.__class__.__name__}")
            if best_params:
                print(f"Best Parameters: {best_params}")
                print(f"Cross-Validation R²: {cv_r2:.4f}")
            elif use_cv:
                print(f"Cross-Validation R²: {cv_r2:.4f}")
            print(f"Train R²: {metrics['r2_train']:.4f}")
            print(f"Validation R²: {metrics['r2_val']:.4f}")
            print(f"Validation RMSE: {metrics['rmse_val']:.4f}")
            print(f"Validation MAE: {metrics['mae_val']:.4f}")

        return row


def split_train_test(df, target_column='2-2.3 nm', test_size=0.2, random_state=40, shuffle=True):
    """
    Splits a DataFrame into train and test sets.

    Parameters:
    - df (pd.DataFrame): The full dataset.
    - target_column (str): The name of the target variable.
    - test_size (float): Proportion of data to use for testing.
    - shuffle (bool): Whether to shuffle before splitting.

    Returns:
    - X_train, X_test, y_train, y_test (tuple): Training and test datasets.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)

    return X_train, X_test, y_train, y_test


def perform_ols(X_train, y_train, X_test, y_test, date_column=None):
    """
    Fits a quick linear regression model using statsmodels.

    Parameters:
    df (pd.DataFrame): The input DataFrame with numerical columns.
    target_column (str): The name of the target variable column.

    Returns:
    sm.OLS: The fitted linear regression model.
    """

    X_train = X_train.select_dtypes(include=[np.number]).dropna()
    X_test = X_test.select_dtypes(include=[np.number]).dropna()
    y_train = y_train[X_train.index]
    y_test = y_test[X_test.index]

    X = sm.add_constant(X_train)

    y = y_train

    model = sm.OLS(y, X).fit()

    return model
