# Wave Energy Forecasting and Storage-Aware Smoothing

**Author:** Chouaib Benchraka

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
│   ├── 01_wave_data_preparation.ipynb
│   ├── 02_wec_power_estimation.ipynb
│   ├── 03_forecasting_baselines.ipynb
│   ├── 04_prediction_intervals_uncertainty.ipynb
│   ├── 05_storage_smoothing_metrics.ipynb
│   └── 06_simses_bess_degradation_demo.ipynb
├── data/                       # Not committed
│   ├── processed/
│   └── raw/
├── outputs/
│   ├── figures/
│   ├── models/                  
│   ├── notebook_03/
│   ├── notebook_04/
│   ├── notebook_05/
│   ├── notebook_06/
│   ├── pdf/                    # PDF rendered notebooks
│   └── tables/
├── references/
│   ├── literature_map.csv
│   └── references.bib
└── src/
```

* `notebooks/` – main analysis and modelling notebooks.
* `data/` – local folder for retrieved and processed data, not committed.
* `outputs/` – generated figures, tables, and notebook outputs.
* `references/` – literature notes and reference tables.
* `src/` – lightweight helper functions used to keep the notebooks readable while preserving the main analysis flow.

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

Data files are not committed to the repository. The wave observations used in the examples are downloaded and prepared in Notebook 01 ([PDF](outputs/pdf/01_wave_data_preparation.pdf) | [Notebook](notebooks/01_wave_data_preparation.ipynb)), which documents the Copernicus Marine / EMODnet source, selected buoy, QC handling, and processed output generation.

---

## Notebooks overview

* *00 – Literature Map: Wave-Energy Forecasting, Uncertainty, and Storage-Aware Smoothing* ([PDF](outputs/pdf/00_literature_map.pdf) | [Notebook](notebooks/00_literature_map.ipynb)):
    Maps the literature motivation behind the repository, connecting wave-energy forecasting, uncertainty estimation, WEC smoothing, and storage-aware BESS/HESS grid-integration metrics.

* *01 – Wave Data Preparation: Copernicus/EMODnet In-Situ Sea-State Data for WEC Power Estimation* ([PDF](outputs/pdf/01_wave_data_preparation.pdf) | [Notebook](notebooks/01_wave_data_preparation.ipynb)):
    Retrieves and prepares Copernicus Marine / EMODnet in-situ wave observations from the Leixões coastal buoy, including QC inspection, 30-minute time-grid alignment, and processed sea-state variables for later WEC power estimation and forecasting.

* *02 – WEC Power Estimation: Leixões Sea-State Characterization and Generic Power Matrix* ([PDF](outputs/pdf/02_wec_power_estimation.pdf) | [Notebook](notebooks/02_wec_power_estimation.ipynb)):
    Explores Leixões sea-state height, period, direction, and spreading variables, then builds a simplified generic power-matrix-style WEC estimate. The resulting estimated WEC power signal is intended for later forecasting, uncertainty, and storage-smoothing notebooks, not as a validated device model.

* *03 – Forecasting Baselines: Short-Term Point Forecasts for Estimated WEC Power* ([PDF](outputs/pdf/03_forecasting_baselines.pdf) | [Notebook](notebooks/03_forecasting_baselines.ipynb)):
    Builds power-only autoregressive forecasting baselines for the estimated WEC power series across 30 min, 1 h, 2 h, and 4 h horizons. Compares persistence, rolling mean, ridge regression, and random forest using chronological rolling-origin folds, test-set skill scores, forecast trajectories, and residual diagnostics to support the following uncertainty-analysis notebook.

* *04 – Prediction Intervals and Uncertainty for Short-Term Estimated WEC Power Forecasts* ([PDF](outputs/pdf/04_prediction_intervals_uncertainty.pdf) | [Notebook](notebooks/04_prediction_intervals_uncertainty.ipynb)):
    Builds empirical prediction intervals from Notebook 03 calibration residuals, compares residual quantile, conformal-style, KDE residual, and global residual interval methods, and evaluates coverage, width, interval score, 250 kW-scale interpretation, and high-ramp behaviour.

* *05 – Storage-Aware Smoothing Metrics for Short-Term Estimated WEC Power* ([PDF](outputs/pdf/05_storage_smoothing_metrics.pdf) | [Notebook](notebooks/05_storage_smoothing_metrics.ipynb)):
    Evaluates simple storage-aware smoothing metrics for the estimated 250 kW WEC proxy. Compares no smoothing, observed-power smoothing, forecast-informed smoothing, and uncertainty-aware lower-bound smoothing using ramp-rate reduction, implied storage power rating, usable energy requirement, throughput, equivalent full cycles, and high-ramp case-study diagnostics.

* *06 – SimSES BESS Degradation Demonstration for WEC Smoothing Profiles* ([PDF](outputs/pdf/06_simses_bess_degradation_demo.pdf), [Notebook](notebooks/06_simses_bess_degradation_demo.ipynb)):
    replays selected WEC storage-smoothing duty cycles from Notebook 05 through a simple [SimSES](https://tum-ees.github.io/simses/) battery workflow. The notebook inspects SOC movement, throughput, equivalent full cycles, C-rate proxies, illustrative thermal response, and degradation-relevant state-of-health indicators. It is framed as a transparent battery-simulation demonstration, not as optimized BESS sizing or validated lifetime prediction.

## AI-assisted development disclosure

This repository was developed by the author with AI-assisted support from ChatGPT for project planning, code refactoring suggestions, Markdown wording, debugging, and consistency review. The author selected the project scope, read and interpreted the literature, chose the modelling workflow, reviewed and edited the generated suggestions, executed the notebooks, checked the outputs, and remains responsible for the final analysis, code, interpretation, and repository content.

