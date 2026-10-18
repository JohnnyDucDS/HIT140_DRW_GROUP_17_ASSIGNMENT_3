# HIT140 Regression 2.2

This project predicts a team's goals in a World Cup 2026 match. It compares OLS, Ridge and Lasso with a mean baseline and a FIFA-points-only model.

## 1. Files

Keep these files in the same folder. Use the Python names below so the imports work.

| File | Purpose |
| --- | --- |
| `build_dataset.py` | Reads Excel, adds opponent data and creates the CSV. |
| `prepare_data.py` | Checks the data, splits it by date and prints summary statistics. |
| `regression_2_2.py` | Fits OLS and saves its results. |
| `plots.py` | Makes the OLS plots and training correlation heatmap. |
| `compare_models.py` | Compares models, tunes Ridge and Lasso, runs paired checks and makes the report figure. |

The input is `data_wrangling_filled (1).xlsx`, sheet `in`. Put it in the same folder. If its name is different, update `EXCEL_PATH` in `build_dataset.py`.

The workbook contains the prepared pre-match statistics and per-90 values. The scripts use those values rather than downloading data.

## 2. Install and run

Open a terminal in this folder. Install the packages:

```bash
python -m pip install pandas numpy "scipy>=1.10" scikit-learn matplotlib openpyxl
```

SciPy 1.10 or later is needed for the t confidence intervals. Install and run using the same Python environment.

Run the scripts in this order:

```bash
python build_dataset.py
python prepare_data.py
python regression_2_2.py
python plots.py
python compare_models.py
```

The first script creates `regression_2_2_restart.csv`. Running the scripts again replaces their saved files.

## 3. Check the data

Expected dataset: 208 rows, 104 matches, 48 teams, eight predictors and no missing values.

| Set | Matches | Rows | Dates |
| --- | ---: | ---: | --- |
| Training | 81 | 162 | Up to 1 July 2026 |
| Test | 23 | 46 | From 2 July 2026 |

Each match has two team rows. Rows from the same date stay together. The target, `act_goals`, includes extra-time goals and excludes penalty shoot-outs.

Validation uses three time splits. Inside each fitting window, Ridge and Lasso use three more time splits to choose alpha. Scaling is fitted only on the fitting rows.

## 4. Find the outputs

- `outputs/`: OLS predictions, scores, coefficients, equation, validation and VIF.
- `outputs/model_comparison/`: comparison scores, alpha searches, Ridge coefficients and equation, and paired checks.
- `outputs/figures/`: five PNG figures.

`comparison_summary.csv` contains mean validation MAE and RMSE. `test_comparison.csv` contains test scores.

`additional_paired_checks.csv` compares Ridge with the baseline and FIFA-only. It contains mean MAE gains, t confidence intervals, and two-sided t-test and Wilcoxon p-values. The checks average the two teams' absolute errors within each match. Positive gains mean lower error for Ridge.

Figures 01 and 02 show OLS predictions and residuals. Figure 03 shows training correlations. Figure 04 contains the Q-Q checks. Figure 05 compares validation RMSE across three folds and shows the Ridge training residual Q-Q plot. Use figure 05 in the report.

## 5. Notes on the results

Ridge had the lowest mean validation RMSE among the eight-input models. FIFA-only was slightly better overall. OLS had the lowest test RMSE.

The test set was inspected during earlier feature experiments, so these results are exploratory. All 23 test matches were knockout games. Repeated teams may make matches dependent, and the Q-Q checks suggested departures from normality. Wilcoxon also assumes symmetric error differences.
