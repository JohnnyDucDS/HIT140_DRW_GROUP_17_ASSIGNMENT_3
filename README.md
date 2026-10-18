# HIT140 Assignment 3 - World Cup 2026 Regression Analysis

This repository contains the data preparation code and two regression analyses for HIT140 - Foundations of Data Science. The analyses use pre-match team statistics to predict goal difference and team goals in the 2026 FIFA World Cup.

Prepared workbooks, modelling datasets and saved outputs are included. After installing the Python packages, you can run either regression directly without collecting the source data again.

## Repository contents

| Folder | Contents |
|---|---|
| [`masterdata/`](masterdata/README.md) | Source-data collection, validation and workbook preparation code, plus the prepared master workbook. |
| [`Assignment3_Regression_2_1/`](Assignment3_Regression_2_1/README.md) | Match-level goal-difference regression, model comparisons, diagnostics and report images. |
| [`Assignment3_Regression_2.2/`](Assignment3_Regression_2.2/README.md) | Team-level goals regression, model comparisons and saved tables and figures. |

Each folder has its own README with more detail.

## Analyses

| Analysis | Target | Dataset | Training and test split |
|---|---|---|---|
| Regression 2.1 | Team A goals minus Team B goals | 104 match rows | 83 development matches and 21 reserved test matches; random split with `random_state=42`. |
| Regression 2.2 | Goals scored by an individual team | 208 team rows from 104 matches | 162 training rows from 81 matches and 46 test rows from 23 matches; split by date. |

Regression 2.1 starts with eight predictors, explores feature reduction and transformations, and compares OLS, Ridge and Random Forest. Its final OLS script uses five predictors. Development comparisons use five-fold cross-validation, including ten repetitions for the main model comparisons.

Regression 2.2 uses eight team and opponent predictors. It compares OLS, Ridge and Lasso with a training-mean baseline and a FIFA-points-only model. Validation follows date order, with regularisation tuned inside the training windows. Both team rows from a match stay in the same set.

## Setup

Use Python 3 and install the packages in the same environment used to run the scripts:

```powershell
python -m pip install numpy pandas "scipy>=1.10" scikit-learn statsmodels matplotlib openpyxl
```

The commands below are intended to be run from the repository root. Each block enters the relevant analysis folder and returns to the root when finished. Keep the folder names, workbook names and local Python module names unchanged.

## Run Regression 2.1

```powershell
cd Assignment3_Regression_2_1

python wrangling.py
python LR_1_1.py
python LR_1_EDA.py
python LR_1_Optimisation.py
python LR_1_TrainingSize.py
python LR_1_Comparison_5folds_10repeats.py
python LR_1_RandomForest_Ridge.py
python LR_1_FinalModel_with_Diagnostics.py

cd ..
```

Input: `Assignment3_Regression_2_1/data_wrangling_filled.xlsx`, sheet `in`.

- `wrangling.py` creates `regression_2_1.csv`.
- `LR_1_1.py` creates `cv_predictions.csv` and `cv_metrics.csv`; run it before EDA.
- EDA saves `images/Figure_2_EDA.png`.
- The repeated-CV comparison saves `images/Figure_3_TwoPanel.png`, showing CV performance and reserved-test predictions.
- The final script saves `LR_1_Actual_vs_Predicted.png` in the analysis folder.
- The EDA heatmap and training-size plots are displayed and need manual saving. Close plot windows to let their scripts finish.

The numbered screenshots in `images/` are retained report outputs. Rerunning the scripts does not automatically replace those screenshots.

## Run Regression 2.2

```powershell
cd Assignment3_Regression_2.2

python build_dataset.py
python prepare_data.py
python regression_2_2.py
python plots.py
python compare_models.py

cd ..
```

Input: `Assignment3_Regression_2.2/data_wrangling_filled (1).xlsx`, sheet `in`.

- `build_dataset.py` creates `regression_2_2_restart.csv`, including the opponent's pre-match inputs.
- `prepare_data.py` checks the dataset and reports the date-based split.
- `regression_2_2.py` saves OLS predictions, evaluation metrics, coefficients, the regression equation, validation results and VIF under `outputs/`.
- `plots.py` creates figures 01-03 under `outputs/figures/`.
- `compare_models.py` saves comparison and tuning tables under `outputs/model_comparison/` and creates figures 04 and 05 under `outputs/figures/`.

The report figure is `outputs/figures/05_report_model_comparison.png`.

## Data preparation

The prepared dataset covers 48 teams, 104 matches and 208 team-match rows. Match statistics come from FIFA, FotMob and ESPN.

Each team's predictor values describe one earlier match:

- A team's opener uses its last pre-tournament friendly, or its most recent friendly with published statistics if the last one has none.
- Later rows use the team's previous World Cup match.
- FIFA ranking points use the last official ranking release available before kickoff.

Count statistics are expressed per 90 minutes. Percentage statistics retain their original scale. Actual goals include extra-time goals and exclude penalty shoot-out goals. The prepared workbook contains supporting sheets for definitions, sources, source-match selection and data checks.

### Optional source-data rebuild

The collection scripts are included for reference and rebuilding. **The original input workbook and collection cache are not included in this submission.** They are separate from the prepared workbooks used by the regression scripts.

With the current settings in `masterdata/common.py`, a full rebuild expects:

```text
input/data_wrangling.xlsx       Original unfilled workbook, sheet "in"
raw_cache/                     Download cache, created during collection
intermediate/                  Collected CSV files
output/                        Rebuilt workbook and completeness summary
```

These paths are relative to the repository root. Once the original input workbook is supplied, the source pipeline runs in this order:

```powershell
python masterdata/collect.py
python masterdata/collect_ranking.py
python masterdata/checks.py
python masterdata/prepare.py
```

Initial collection requires internet access and available provider APIs. The rebuilt workbook is written to `output/data_wrangling_filled.xlsx`. To use it in the regressions, replace each analysis folder's workbook copy using its expected filename, then rerun the modelling pipeline. See the [data-preparation guide](masterdata/README.md) for the collection method and original project layout.

## Reproducing and interpreting the results

- Run Regression 2.1 from its analysis directory because it uses relative input and output paths. Keep CSV row order unchanged to preserve its fixed split.
- After changing an input workbook, rebuild the corresponding modelling CSV and rerun the affected analysis. Generated CSVs and automatically exported figures are overwritten.
- Comment-only edits do not change the calculations.
- Saved results belong to the included data and model settings. The two regressions use different targets and evaluation splits.
- Regression 2.2's test set was inspected during earlier feature experiments, so its final comparisons are exploratory. Its folder README records the evaluation limitations in more detail.
