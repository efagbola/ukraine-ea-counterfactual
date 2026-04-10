# Ukraine Euro-Area Counterfactual

This repository contains my final exam submission for Quantitative Methods in Finance (QMF). The project estimates a structural VAR counterfactual for Ukraine’s inflation under hypothetical Euro Area membership, using a Blanchard–Quah identification strategy and regime-dependent treatment intensity motivated by Part A.


## Repository structure
- `Counterfactual_report.pdf` — includes the answer to Part A, as well as the Part B counterfactual figure and interpretation
- `counterfactual.ipynb` — main reproducible notebook  
- `src/model_utils.py` — econometric helper functions  
- `data/` — raw input datasets used by the notebook  
- `output/` — generated figures and derived datasets  

## How to run

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook counterfactual.ipynb
