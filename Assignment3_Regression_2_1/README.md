# Regression 2.1 — Execution Guide

Open a terminal **in this folder** before running the scripts. Keep `data_wrangling_filled.xlsx` here.

Install the required Python packages if needed:

```powershell
python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib openpyxl
```

Run the files in this order using `python filename.py`:

| Order | File | Purpose |
|---|---|---|
| 1 | `wrangling.py` | Reads the workbook and creates `regression_2_1.csv`. |
| 2 | `LR_1_1.py` | Fits the initial eight-predictor model and creates `cv_predictions.csv` and `cv_metrics.csv`. |
| 3 | `LR_1_EDA.py` | Reads the CV predictions and produces correlations, regression diagnostics and plots. |
| 4 | `LR_1_Optimisation.py` | Compares predictor combinations and transformations using five-fold validation. |
| 5 | `LR_1_TrainingSize.py` | Examines how training size affects model performance. |
| 6 | `LR_1_Comparison_5folds_10repeats.py` | Compares the original, reduced and transformed models over repeated validation. |
| 7 | `LR_1_RandomForest_Ridge.py` | Compares OLS, Ridge and Random Forest. |
| 8 | `LR_1_FinalModel_with_Diagnostics.py` | Evaluates the selected five-predictor OLS model and prints its final diagnostics and test predictions. |

For example, start with:

```powershell
python wrangling.py
python LR_1_1.py
python LR_1_EDA.py
```

- After changing the workbook, rerun steps 1 and 2 before the remaining analysis so the CSVs stay current. These steps overwrite their generated CSVs.
- Step 3 requires the CV predictions from step 2. Steps 4–8 read the modelling CSV from step 1 independently; the order above follows the analysis sequence.
- Close plot windows to let a script finish before starting the next one.
- `images/` contains saved screenshots and is **not automatically refreshed**. Most plots need manual saving. Step 8 also saves `LR_1_Actual_vs_Predicted.png` in this folder.

### Reproducibility

All modelling scripts use the same fixed random seed (`random_state=42`) and an 80/20 train-test split. The 21 reserved test matches are excluded from cross-validation and model optimisation. Keep the input dataset in its original row order to reproduce the reported results.

The scripts generate their output files automatically where specified. The `images/` folder contains figures used in the report and may require manual updates after rerunning the analyses.
