import hydra
import mlflow
from omegaconf import OmegaConf
from utils.hydra_utils import prepare_data_hydra
from utils.hydra_modeling import perform_regression_mlflow_hydra
import hydra.core.hydra_config


@hydra.main(config_path="../config", config_name="varrio_sweeper", version_base="1.3")
def main(cfg):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    sweep_run_idx = hydra_cfg.job.num if hydra_cfg.job.num is not None else "N/A"

    print(f"\n🔄 Running Hydra Experiment #{sweep_run_idx}")
    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, y_train, X_test, y_test = prepare_data_hydra(cfg)

    results_df = perform_regression_mlflow_hydra(
        cfg, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
