# Ukraine Euro-Area Counterfactual

This repository contains my final exam submission for QMF. The project estimates a structural VAR counterfactual for Ukraine’s inflation under hypothetical Euro Area membership using a Blanchard-Quah identification strategy and regime-dependent treatment intensity motivated by the work done in Part A.

## Repository structure

- `counterfactual.ipynb` — main reproducible notebook
- `src/model_utils.py` — reusable econometric helper functions
- `data/` — raw input datasets used by the notebook
- `output/` — generated figures and derived datasets

## How to run

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook counterfactual.ipynb
