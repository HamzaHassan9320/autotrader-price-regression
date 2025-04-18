# AutoTrader Price Regression

**Short summary**

Predicting second‑hand car prices with classic tabular ML.  
Data: 402 006 rows · 12 columns (target = `price`).  
Models compared: Linear Regression · Random Forest · Gradient Boosting · Voting Ensemble.

---

## Table of contents
1. [Project motivation](#project-motivation)
2. [Data](#data)
3. [Quick start](#quick-start)
4. [Notebook & code guide](#notebook--code-guide)
5. [Results at a glance](#results-at-a-glance)
6. [Model interpretation](#model-interpretation)
7. [Directory layout](#directory-layout)
8. [License](#license)

---

## Project motivation
Buying a used car is a price‑sensitive decision.  
The goal here is to build transparent, reproducible baselines that predict `price`
given mileage, age, fuel‑type and a handful of categorical descriptors.  
Grades in the coursework are **not** the focus; the code and discussion are.

## Data
* Source: AutoTrader public adverts (exported as `Adverts.csv`).
* Rows: 402 006 *Columns*: 12 (all but `price` used as predictors).
* Basic cleaning:
  * Remove mileage & price outliers via 1.5 × IQR.  
  * Drop cars registered before 1975.  
  * Mode‑impute categorical gaps in `fuel_type`, `body_type`, `standard_colour`. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
* Two engineered features:
  * `vehicle_age` (`2024 – year_of_registration`)
  * `mileage_to_age_ratio` (`mileage / vehicle_age`) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

See `notebooks/01_explore.ipynb` for the exact steps.

## Quick start
```bash
# clone & create environment
git clone https://github.com/<your‑user>/autotrader-price-regression.git
cd autotrader-price-regression
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate autotrader-price

# run the full pipeline
python src/train.py --config configs/base.yaml
