# Hydra Configuration Reference

This document describes the Hydra configuration structure and parameters used in the Negative Ions project.

---

## 📁 Config Structure

The configurations are structured using Hydra’s config composition. Each experiment YAML file inherits from a shared base config (`base_varrio.yaml`) and selectively overrides relevant fields.

- **Base config**: `base_varrio.yaml`
- **Experiment examples**: `varrio_methods.yaml`, `varrio_sweeper.yaml`, etc.

Change in file ```src/hydra_runner.py```
the line
```@hydra.main(config_path="../config", config_name="varrio_sweeper", version_base="1.3")```
to correspond to relevant experiment name. 

You can run an experiment with:
```bash
python hydra_runner.py --multirun
```

---

## 🔧 Top-Level Parameters

### `experiment`
| Field | Description |
|-------|-------------|
| `name` | Name for the experiment (used in MLflow and logs) |

---

### `data`
| Field | Description |
|-------|-------------|
| `file_path` | Path to the dataset (CSV format) |
| `target` | Column name for the target variable (e.g. "2-2.3 nm") |
| `timestamp` | Column containing timestamps (e.g. "Datetime") |
| `year`, `years` | Optional filters for specific years |
| `start_date`, `end_date` | Filter window (e.g. "07-01" to "08-31") |
| `drop_cols` | Columns to drop from the dataset |
| `permute_target_random` | Randomize target variable across whole dataset |
| `permute_target_day` | Randomize target based on full days |

---

### `split`
| Field | Description |
|-------|-------------|
| `year_split` | Use full years for test/train split |
| `test_years` | List of years to use for testing |
| `test_size` | Test set fraction for random split |
| `random_state` | Random seed |
| `shuffle` | Shuffle rows before split |
| `day_split` | Use day-wise splitting logic |

---

### `preprocessing.missing_value_strategy`

Controls how missing values are handled.

```yaml
missing_value_strategy:
  selected: "dropna"
  options:
    dropna: {}
    interpolation: 
      method: "linear"
      linear_features: [...]
      circular_features: [...]
```

- **dropna**: Drop rows with any missing value
- **interpolation**: Interpolate missing values using:
  - `linear_features`: time-series linear interpolation
  - `circular_features`: angular data (e.g., wind direction)

---

### `preprocessing.transformations`

| Field | Description |
|-------|-------------|
| `add_year` | Add year as a feature |
| `wind_method` | Method to convert wind direction (e.g. `xy`) |
| `add_time` | Add hour-of-day and day-of-week as cyclic features |
| `period_method` | How to encode periodic features (e.g., `cyclic`) |
| `add_h2o` | Add water vapor estimate |
| `h2o_method` | Method to calculate H₂O (e.g. `goff_gratch`) |
| `filter_rain` | Drop rows based on rain intensity |
| `rain_column`, `rain_threshold` | Column and threshold for rain filtering |
| `filter_par` | Drop rows based on PAR |
| `par_column`, `par_threshold` | Column and threshold for PAR filtering |
| `add_time_lags` | Enable/disable lagged features |
| `time_lagged_features` | Dictionary: feature → list of lags |

---

### `preprocessing.normalizations`

| Type | Description |
|------|-------------|
| `log_features` | Apply log(x+1) transform |
| `minmax_features` | Scale to [0, 1] |
| `standard_features` | Standardize to zero mean, unit variance |
| `quantile_features` | Quantile transform (optional) |

---

### `preprocessing.polynomial_features`

| Field | Description |
|-------|-------------|
| `degree` | Degree of polynomial features (null = disabled) |

---

### `models`

A list of model configurations to run.

Each model has:
- `name`: sklearn-compatible regressor
- `params`: dictionary of hyperparameters

**Example:**
```yaml
- name: "LGBMRegressor"
  params:
    learning_rate: 0.01
    max_depth: 8
    ...
```

---

### `training`

| Field | Description |
|-------|-------------|
| `use_cv` | Enable cross-validation |
| `cv_folds` | Number of folds |
| `scoring` | Scoring metric (e.g., "r2") |
| `search_method` | Hyperparameter search ("grid", "random") |
| `n_iter` | Iterations for random search |
| `use_time_series_cv` | Use time-series-aware cross-validation |
| `use_groupKFold` | Use GroupKFold split logic |

---

### `mlflow`

| Field | Description |
|-------|-------------|
| `experiment_name` | Name of MLflow experiment |
| `tracking_uri` | Backend URI (e.g., SQLite database) |

---

### `hydra.sweeper.params`

Optional parameters for multirun sweeps. In case multiple sweeped parameters are specified hydra runs all combinations. 

```yaml
hydra:
  sweeper:
    params:
      split.test_years: 
        choice([2019], [2020], [2021], [2022])
```

Use via:
```bash
python hydra_runner.py --multirun
```

---

## 📝 Notes

- All experiment configs inherit from `base_varrio.yaml` by default
- Use `OmegaConf.to_yaml(cfg)` in your runner to inspect full config at runtime
- More detailes explanation of parameters is available in [Thesis](Mikko_Ahro_Mastersthesis_2025.pdf)

---


