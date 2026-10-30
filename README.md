# REFIT Energy Disaggregation Project

## Objective
Cleaning REFIT household electricity data, exploring washing-machine usage, and estimating appliance-level power consumption from aggregate electricity readings.

## Pipeline
1. Raw House 1 data cleaning and auditing.
2. One-minute CLEAN REFIT data preparation.
3. Washing-machine exploratory data analysis and cycle detection.
4. NILM modelling using HistGradientBoostingRegressor.
5. Evaluation at 15-minute and 30-minute resolutions.

## Models
- Always-off baseline
- Hour-of-week profile
- Pooled HistGradientBoostingRegressor
- Leave-one-house-out evaluation

## Key Results
Pooled model test MAE: 28.229 W
Pooled model test RMSE: 156.296 W
Pooled model ON-state F1: 0.333

## Limitations
The model uses a centred aggregate-power window and therefore uses future readings. It is an offline estimation approach, not a real-time causal model.

## Repository Structure
- src/: Python scripts
- outputs/: Audit reports, EDA, model results and plots
- README.md: Project overview

## Data
REFIT dataset is not included. Refer to DATA.md for dataset information.
