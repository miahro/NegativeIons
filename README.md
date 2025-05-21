# **Negative Ions**

## **Description**

Description here

## **Table of Contents**
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

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

## **Installing New Packages to Conda Environment**

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
