# Wave Energy Forecasting and Storage-Aware Smoothing

This repository contains a small set of Jupyter notebooks exploring short-term wave-energy forecasting and storage-aware smoothing.

The focus is on working with open wave-resource data, estimating simplified wave energy converter power output, evaluating forecasting and uncertainty methods, and connecting the results to BESS/HESS grid-integration metrics.

---

## Repository structure

```text
wave-energy-forecasting-storage-ml/
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
├── notebooks/
│   └── 00_literature_map.ipynb
├── data/                       # Not committed
├── outputs/
│   └── pdf/                    # PDF render notebooks
├── references/
│   ├── literature_map.csv
│   └── references.bib
└── src/
```

* `notebooks/` – main analysis and modelling notebooks.
* `data/` – local folder for datasets, not committed.
* `outputs/` – generated figures, tables, and notebook outputs.
* `references/` – literature notes and reference tables.
* `src/` – optional helper functions.

---

## Conda environment

The environment is defined in `environment.yml`.

The environment was created using:

```text
conda 26.3.2
```

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate wave-energy-storage-ml
```

The environment installs MHKiT from its GitHub source repository, so `git` is included as an environment dependency.

A `requirements.txt` file is also provided as a package-version snapshot of the working environment. It is mainly included for transparency and reproducibility; `environment.yml` remains the recommended installation file.

---

## Data


---

## Notebooks overview

* 00 – Literature Map: Wave-Energy Forecasting, Uncertainty, and Storage-Aware Smoothing ([PDF](notebooks/pdf/00_literature_map.pdf) | [Notebook](notebooks/00_literature_map.ipynb))

  Maps the literature motivation behind the repository, connecting BESS/grid-integration context with wave-energy forecasting, uncertainty estimation, WEC smoothing, and BESS/HESS relevance.


