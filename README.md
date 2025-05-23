# **Negative Ions**

## **Description**

Description here

## **Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- 
- [License](#license)


---

## **Features** 
- Feature 1: Example description
- Feature 2: Example description
- Feature 3: Example description

---

## **Installation**

### Initial Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:your-username/your-repo-name.git
cd your-repo-name
```

1. Pull the latest version of the repo:
```bash
git pull origin main
```

2. Create and activate the Conda environment:
```bash
conda create -n negion python=3.10
conda activate negion
```

### Installing Updated Dependencies

To update the environment based on `environment.yml`:

```bash
conda env update --file environment.yml --prune
```

---

### **Installing New Packages to Conda Environment**

1. Make sure the `negion` environment is active:
```bash
conda activate negion
```

2. Install the new package:
```bash
conda install <library-name>
```

3. After installation, update `environment.yml`:
```bash
conda env export --no-builds | grep -v "^prefix:" > environment.yml
```

4. Commit and push the updated `environment.yml`:
```bash
git add environment.yml
git commit -m "Update Conda environment"
git push origin main
```

---


## File and variable configuration

The locations of kocal files (downloaded SMEAR data, target variable as text files, metadata and intermediate preprocessing results) are defined in file ```file_config.py```. Adjust as necessary.

The parameters to be downloaded from SmartSMEAR API are defined in Jupyter notebook. ```smear_loader.ipynb```, adjust as necessary.


## Loading, proprocessing and exploring the data

1. Negative ion concentration data is assumed to be text file, with two columns: timestamp and concentration. The location of the file is specified in file ```file_config.py```
2. Define SMEAR parameters to be downloaded in Jupyter notebook ```smear_loader.ipynb``` and run the notebook. Note, the downloaded data is saved into local ```.csv``` file, you only need to run loader once, unless downloaded parameters are changed.
3. Make data preprocessing (combines SMEAR paramters and target variable) by running notebook ```preprocessing.ipynb```. Note, combined data is saved into local ```.csv```file, you only need to run preprocessing notebook once, unless set of parameters is changed. 
4. Perform exploratory analysis by running notebook ```varrio_exploratory.ipynb```. Add statistical tests as necessary. Currently exploratory analysis is performed only for Värriö data, but other stations can be analyzed by copying notebook, and changing file and variable references. 


---




## **MLflow**

To start an MLflow tracking server locally:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
```

> 💡 Ensure `mlflow` is installed in your Conda environment via conda-forge:
> ```bash
> conda install -c conda-forge mlflow
> ```

---




## **License**

[License]()
