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
│   ├── 00_literature_map.ipynb
│   └── 01_wave_data_preparation.ipynb
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

Data files are not committed to the repository. The wave observations used in the examples are downloaded and prepared in [`01_wave_data_preparation.ipynb`](notebooks/01_wave_data_preparation.ipynb), which documents the Copernicus Marine / EMODnet source, selected buoy, QC handling, and processed output generation.

---

## Notebooks overview

* 00 – Literature Map: Wave-Energy Forecasting, Uncertainty, and Storage-Aware Smoothing ([PDF](outputs/pdf/00_literature_map.pdf) | [Notebook](notebooks/00_literature_map.ipynb))

  Maps the literature motivation behind the repository, connecting BESS/grid-integration context with wave-energy forecasting, uncertainty estimation, WEC smoothing, and BESS/HESS relevance.

* 01 – Wave Data Preparation: Copernicus/EMODnet In-Situ Sea-State Data for WEC Power Estimation ([PDF](outputs/pdf/01_wave_data_preparation.pdf) | [Notebook](notebooks/01_wave_data_preparation.ipynb))

  Retrieves and prepares Copernicus Marine / EMODnet in-situ wave observations from the Leixões coastal buoy, including QC inspection, 30-minute time-grid alignment, and processed sea-state variables for later WEC power estimation and forecasting.

