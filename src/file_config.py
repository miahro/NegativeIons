import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."))  # Moves one level up
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
NEG_IONS_DIR = os.path.join(RAW_DIR, "negative ions")
EXPLORATORY_DIR = os.path.join(DATA_DIR, "exploratory")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# File Paths Dictionary
FILE_PATHS = {

    # Raw Data Files
    "var_raw": os.path.join(RAW_DIR, "var_raw_data.csv"),
    "hyy_raw": os.path.join(RAW_DIR, "hyy_raw_data.csv"),
    "sii_raw": os.path.join(RAW_DIR, "sii_raw_data.csv"),
    "kum_raw": os.path.join(RAW_DIR, "kum_raw_data.csv"),
    "var_rain": os.path.join(RAW_DIR, "var_rain.csv"),
    # "var_raw_w_rain": os.path.join(RAW_DIR, "var_raw_w_rain.csv"),
    # "hyy_raw_w_rain": os.path.join(RAW_DIR, "hyy_raw_w_rain.csv"),

    # Negative Ions Source Data
    "var_neg_ions_txt": os.path.join(NEG_IONS_DIR, "Varrio negaaive ions.txt"),
    "hyy_neg_ions_txt": os.path.join(NEG_IONS_DIR, "Hyytiala negaaive ions.txt"),
    "sii_neg_ions_txt": os.path.join(NEG_IONS_DIR, "Siikaneva negaaive ions.txt"),
    "kum_neg_ions_txt": os.path.join(NEG_IONS_DIR, "Kumpula negaaive ions.txt"),

    # Negative ion concenstraion converted CSV Files
    "var_neg_ions_csv": os.path.join(RAW_DIR, "var_neg_ions.csv"),
    "hyy_neg_ions_csv": os.path.join(RAW_DIR, "hyy_neg_ions.csv"),
    "sii_neg_ions_csv": os.path.join(RAW_DIR, "sii_neg_ions.csv"),
    "kum_neg_ions_csv": os.path.join(RAW_DIR, "kum_neg_ions.csv"),

    # Metadata CSVs
    "var_metadata": os.path.join(RAW_DIR, "var_metadata.csv"),
    "hyy_metadata": os.path.join(RAW_DIR, "hyy_metadata.csv"),
    "sii_metadata": os.path.join(RAW_DIR, "sii_metadata.csv"),
    "kum_metadata": os.path.join(RAW_DIR, "kum_metadata.csv"),

    # Exploratory Data Files
    "var_metadata_range": os.path.join(EXPLORATORY_DIR, "var_metadata_range.csv"),
    "hyy_metadata_range": os.path.join(EXPLORATORY_DIR, "hyy_metadata_range.csv"),
    "sii_metadata_range": os.path.join(EXPLORATORY_DIR, "sii_metadata_range.csv"),
    "kum_metadata_range": os.path.join(EXPLORATORY_DIR, "kum_metadata_range.csv"),
    "var_combined": os.path.join(EXPLORATORY_DIR, "var_combined.csv"),
    "hyy_combined": os.path.join(EXPLORATORY_DIR, "hyy_combined.csv"),
    "sii_combined": os.path.join(EXPLORATORY_DIR, "sii_combined.csv"),
    "kum_combined": os.path.join(EXPLORATORY_DIR, "kum_combined.csv"),


    # 'var_summer_csv': os.path.join(EXPLORATORY_DIR, 'var_summer.csv'),
    # "var_summer_w_rain_csv": os.path.join(EXPLORATORY_DIR, "var_summer_w_rain.csv"),
    # "var_all_w_rain_csv": os.path.join(EXPLORATORY_DIR, "var_all_w_rain.csv"),


    # Intermediate results
    'var_lin_reg_pol': os.path.join(RESULTS_DIR, 'var_lin_reg_pol.csv'),
    'var_lin_reg_normalized': os.path.join(RESULTS_DIR, 'var_lin_reg_normalized.csv'),
    'var_dt': os.path.join(RESULTS_DIR, 'var_dt.csv'),
    'var_rf': os.path.join(RESULTS_DIR, 'var_rf.csv'),
    'var_xgboost': os.path.join(RESULTS_DIR, 'var_xgboost.csv'),
    'var_lgbm': os.path.join(RESULTS_DIR, 'var_lgbm.csv'),
    'var_svr': os.path.join(RESULTS_DIR, 'var_svr.csv'),
    'var_nn': os.path.join(RESULTS_DIR, 'var_nn.csv'),
    'var_all_models': os.path.join(RESULTS_DIR, 'var_all_models.csv'),
    'var_all_models_feat': os.path.join(RESULTS_DIR, 'var_all_models_feat.csv'),
    'var_all_models_feat_imp': os.path.join(RESULTS_DIR, 'var_all_models_feat_imp.csv'),
}
