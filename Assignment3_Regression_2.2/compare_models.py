"""Compare models for Regression 2.2.
Ridge and Lasso use nested validation in date order to choose alpha.
The script also evaluates the existing test set and creates report figures.
Test results are exploratory because this set was inspected earlier."""

import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import BASE_DIR, df, features, X_train, y_train
from regression_2_2 import evaluate


# Fix these grids before comparing models. Larger alpha means more shrinkage.
RIDGE_ALPHAS = np.logspace(-3, 4, 15)
LASSO_ALPHAS = np.logspace(-4, 1, 11)


def date_splits(X, n_splits=3):
    """Return row positions; keep both teams and each whole date together."""
    row_dates = df.loc[X.index, "Date"]
    unique_dates = np.array(sorted(row_dates.unique()))
    splits = []

    for fit_dates, validation_dates in TimeSeriesSplit(
        n_splits=n_splits
    ).split(unique_dates):
        fit = np.flatnonzero(row_dates.isin(unique_dates[fit_dates]).to_numpy())
        validation = np.flatnonzero(
            row_dates.isin(unique_dates[validation_dates]).to_numpy()
        )

        fit_rows = df.loc[X.iloc[fit].index]
        validation_rows = df.loc[X.iloc[validation].index]
        if fit_rows["Date"].max() >= validation_rows["Date"].min():
            raise ValueError("Validation must follow the fitting dates.")
        if set(fit_rows["MatchNumber"]) & set(validation_rows["MatchNumber"]):
            raise ValueError("A match appears on both sides of a split.")
        splits.append((fit, validation))

    return splits


def tune_regularized(name, X_fit, y_fit, inner_splits, fold):
    """Choose alpha within this outer fold's fitting data, then refit it."""
    if name == "Ridge (8 inputs)":
        estimator = Ridge()
        alphas = RIDGE_ALPHAS
    else:
        estimator = Lasso(max_iter=100000, tol=1e-6)
        alphas = LASSO_ALPHAS

    # The scaler is fitted separately inside each inner fitting split.
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", estimator),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={"model__alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=inner_splits,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X_fit, y_fit)

    # Retain all candidates, rather than only the selected alpha.
    candidate_rows = []
    for position, alpha in enumerate(search.cv_results_["param_model__alpha"]):
        row = {
            "Outer fold": fold,
            "Model": name,
            "Alpha": float(alpha),
            "Mean inner RMSE": -search.cv_results_["mean_test_score"][position],
            "Selected": position == search.best_index_,
        }
        for inner_fold in range(len(inner_splits)):
            row[f"Inner fold {inner_fold + 1} RMSE"] = -search.cv_results_[
                f"split{inner_fold}_test_score"
            ][position]
        candidate_rows.append(row)

    return (
        search.best_estimator_,
        float(search.best_params_["model__alpha"]),
        candidate_rows,
    )


def main():
    print("MODEL COMPARISON 2.2: TRAINING DATA ONLY")
    print("Training matches:", df.loc[X_train.index, "MatchNumber"].nunique())
    print("Training rows:", len(X_train))
    print("Outer validation folds: 3 (date order)")
    print("Inner tuning folds: 3 (date order, within each outer fit)")
    print("Test set: not evaluated in this step")

    fold_rows = []
    tuning_rows = []
    model_names = [
        "Baseline (training mean)",
        "FIFA points only",
        "OLS (8 inputs)",
        "Ridge (8 inputs)",
        "Lasso (8 inputs)",
    ]

    # These are the same three outer folds used by regression_2_2.py.
    for fold, (fit, validation) in enumerate(date_splits(X_train), start=1):
        X_fit, y_fit = X_train.iloc[fit], y_train.iloc[fit]
        X_validation, y_validation = (
            X_train.iloc[validation], y_train.iloc[validation]
        )
        inner_splits = date_splits(X_fit)
        fit_ids = df.loc[X_fit.index]
        validation_ids = df.loc[X_validation.index]

        for name in model_names:
            alpha = np.nan
            columns = features
            if name == "Baseline (training mean)":
                model = DummyRegressor(strategy="mean").fit(X_fit, y_fit)
                input_count = 0
            elif name == "FIFA points only":
                columns = ["fifa_ranking_points"]
                model = LinearRegression().fit(X_fit[columns], y_fit)
                input_count = 1
            elif name == "OLS (8 inputs)":
                model = LinearRegression().fit(X_fit, y_fit)
                input_count = len(features)
            else:
                model, alpha, candidates = tune_regularized(
                    name, X_fit, y_fit, inner_splits, fold
                )
                tuning_rows.extend(candidates)
                input_count = len(features)

            predictions = model.predict(X_validation[columns])
            scores = evaluate(y_validation, predictions)
            fold_rows.append({
                "Model": name,
                "Fold": fold,
                "Inputs": input_count,
                "Fit matches": fit_ids["MatchNumber"].nunique(),
                "Validation matches": validation_ids["MatchNumber"].nunique(),
                "Last fit date": fit_ids["Date"].max().date(),
                "First validation date": validation_ids["Date"].min().date(),
                "Last validation date": validation_ids["Date"].max().date(),
                "Alpha": alpha,
                **scores,
            })
        print(f"Completed outer fold {fold}/3", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    summary = fold_results.groupby("Model", sort=False).agg(
        **{"Mean MAE": ("MAE", "mean"), "Mean RMSE": ("RMSE", "mean")}
    )
    fold_rmse = fold_results.pivot(index="Model", columns="Fold", values="RMSE")
    fold_rmse.columns = [f"Fold {fold} RMSE" for fold in fold_rmse.columns]
    summary = summary.join(fold_rmse).sort_values(["Mean RMSE", "Mean MAE"])
    summary = summary[[
        "Fold 1 RMSE", "Fold 2 RMSE", "Fold 3 RMSE", "Mean MAE", "Mean RMSE"
    ]]

    output_dir = BASE_DIR / "outputs" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "comparison_summary.csv", index_label="Model")
    fold_results.to_csv(output_dir / "comparison_folds.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "alpha_search.csv", index=False)

    print("\nVALIDATION RESULTS (lower RMSE is better)")
    print(summary.round(4).to_string())
    print("\nSELECTED ALPHAS BY OUTER FOLD")
    print(fold_results.dropna(subset=["Alpha"])[
        ["Model", "Fold", "Alpha"]
    ].to_string(index=False))
    print("\nLowest mean validation RMSE:", summary.index[0])
    eight_input_models = [
        name for name in summary.index if name.endswith("(8 inputs)")
    ]
    print("Lowest mean RMSE among 8-input models:", eight_input_models[0])
    print("Mean RMSE is the equal-weight average of the three outer-fold RMSEs.")
    print("The folds are time windows, not independent statistical replicates.")
    print("Dataset and feature choices were explored earlier; this is a further comparison.")
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()



# STEP 2: Evaluate models on the existing test set
if __name__ == "__main__":
    from prepare_data import X_test, y_test, id_columns

    # Ridge was preferred among 8-input models using training validation.
    selected_name = "Ridge (8 inputs)"
    full_training_splits = date_splits(X_train)

    # Choose alpha using all training data, without using test scores.
    ridge_model, ridge_alpha, ridge_search = tune_regularized(
        "Ridge (8 inputs)",
        X_train,
        y_train,
        full_training_splits,
        "Full training",
    )

    lasso_model, lasso_alpha, lasso_search = tune_regularized(
        "Lasso (8 inputs)",
        X_train,
        y_train,
        full_training_splits,
        "Full training",
    )

    test_models = {
        "Baseline (training mean)": (
            DummyRegressor(strategy="mean").fit(X_train, y_train),
            features,
            "pred_baseline",
        ),
        "FIFA points only": (
            LinearRegression().fit(
                X_train[["fifa_ranking_points"]], y_train
            ),
            ["fifa_ranking_points"],
            "pred_fifa_only",
        ),
        "OLS (8 inputs)": (
            LinearRegression().fit(X_train, y_train),
            features,
            "pred_ols",
        ),
        "Ridge (8 inputs)": (
            ridge_model,
            features,
            "pred_ridge",
        ),
        "Lasso (8 inputs)": (
            lasso_model,
            features,
            "pred_lasso",
        ),
    }

    test_predictions = df.loc[
        X_test.index, id_columns + ["act_goals"]
    ].copy()

    test_scores = {}

    for name, (fitted_model, columns, prediction_column) in test_models.items():
        predicted = fitted_model.predict(X_test[columns])
        test_predictions[prediction_column] = predicted
        test_scores[name] = evaluate(y_test, predicted)

    test_metrics = pd.DataFrame(test_scores).T
    test_metrics.index.name = "Model"
    test_metrics["Selected by validation (8 inputs)"] = (
        test_metrics.index == selected_name
    )
    test_metrics["Alpha"] = [
        np.nan, np.nan, np.nan, ridge_alpha, lasso_alpha
    ]

    # Convert Ridge coefficients back to the original variable units.
    scaler = ridge_model.named_steps["scale"]
    fitted_ridge = ridge_model.named_steps["model"]

    original_coefficients = fitted_ridge.coef_ / scaler.scale_
    original_intercept = (
        fitted_ridge.intercept_
        - np.dot(original_coefficients, scaler.mean_)
    )

    ridge_coefficients = pd.DataFrame({
        "Variable": ["Intercept"] + features,
        "Coefficient": [original_intercept] + list(original_coefficients),
    })

    equation = f"Predicted goals = {original_intercept:.6f}"
    for variable, coefficient in zip(features, original_coefficients):
        equation += f" {coefficient:+.6f} * {variable}"

    # Save the additional results.
    test_output_dir = BASE_DIR / "outputs" / "model_comparison"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    test_metrics.to_csv(test_output_dir / "test_comparison.csv")
    test_predictions.to_csv(
        test_output_dir / "test_model_predictions.csv", index=False
    )
    ridge_coefficients.to_csv(
        test_output_dir / "selected_model_coefficients.csv", index=False
    )
    pd.DataFrame(ridge_search + lasso_search).to_csv(
        test_output_dir / "full_training_alpha_search.csv", index=False
    )
    (test_output_dir / "selected_model_equation.txt").write_text(
        equation, encoding="utf-8"
    )

    print("\nTEST COMPARISON: EXISTING TEST SET")
    print("Training matches:", df.loc[X_train.index, "MatchNumber"].nunique())
    print("Test matches:", df.loc[X_test.index, "MatchNumber"].nunique())
    print("Preferred 8-input model from validation:", selected_name)
    print("Selected Ridge alpha:", ridge_alpha)
    print("Selected Lasso alpha:", lasso_alpha)

    print(test_metrics[
        ["MAE", "MSE", "RMSE", "NRMSE (%)", "R2"]
    ].round(4).to_string())

    print("\nRIDGE COEFFICIENTS: ORIGINAL UNITS")
    print(ridge_coefficients.round(6).to_string(index=False))

    print("\nThis test set was inspected during earlier exploration.")
    print("These results are exploratory, not a new independent test.")
    print("Results saved to:", test_output_dir)



# STEP 3: Compare Ridge and baseline errors by match
if __name__ == "__main__":
    from scipy import stats

    baseline_error = (
        test_predictions["act_goals"]
        - test_predictions["pred_baseline"]
    ).abs()

    ridge_error = (
        test_predictions["act_goals"]
        - test_predictions["pred_ridge"]
    ).abs()

    # Average the two teams' error differences within each match.
    match_gain = pd.DataFrame({
        "MatchNumber": test_predictions["MatchNumber"],
        "MAE_gain": baseline_error - ridge_error,
    }).groupby("MatchNumber")["MAE_gain"].mean()

    test_result = stats.ttest_1samp(match_gain, popmean=0)
    confidence_interval = test_result.confidence_interval(
        confidence_level=0.95
    )

    inference_results = pd.DataFrame([{
        "Matches": len(match_gain),
        "Mean MAE gain": match_gain.mean(),
        "95% CI lower": confidence_interval.low,
        "95% CI upper": confidence_interval.high,
        "t statistic": test_result.statistic,
        "df": len(match_gain) - 1,
        "p value (two-sided)": test_result.pvalue,
    }])

    inference_results.to_csv(
        test_output_dir / "ridge_vs_baseline_test.csv",
        index=False
    )

    print("\nSTEP 3: RIDGE VS BASELINE")
    print("Positive MAE gain means lower error for Ridge.")
    print(inference_results.round(4).T.to_string(header=False))
    print("Exploratory test; assumes independent matches.")



# STEP 4: Q-Q checks
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats

    training_residuals = y_train - ridge_model.predict(X_train)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    stats.probplot(training_residuals, dist="norm", plot=axes[0])
    axes[0].set_title(
        f"Ridge training residuals (n={len(training_residuals)})"
    )
    axes[0].set_ylabel("Ordered residuals (goals)")

    stats.probplot(match_gain, dist="norm", plot=axes[1])
    axes[1].set_title(
        f"Match-level MAE gain (n={len(match_gain)})"
    )
    axes[1].set_ylabel("Ordered MAE gains (goals)")

    for ax in axes:
        ax.set_xlabel("Theoretical normal quantiles")
        ax.grid(alpha=0.2)

    figure_dir = BASE_DIR / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    figure_path = figure_dir / "04_qq_checks.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSTEP 4: Q-Q FIGURE SAVED")
    print(figure_path)



# STEP 5: Reproduce the figure used in the report
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats

    report_dir = BASE_DIR / "outputs" / "model_comparison"
    report_cv = pd.read_csv(
        report_dir / "comparison_summary.csv", index_col="Model"
    )
    report_test = pd.read_csv(
        report_dir / "test_comparison.csv", index_col="Model"
    )

    model_names = [
        "Baseline (training mean)",
        "FIFA points only",
        "OLS (8 inputs)",
        "Ridge (8 inputs)",
        "Lasso (8 inputs)",
    ]
    model_labels = [
        "Mean baseline", "FIFA points only", "OLS", "Ridge", "Lasso"
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.25, 2.55),
        gridspec_kw={"width_ratios": [1.12, 1]}
    )

    # Panel A: RMSE across validation folds
    report_folds = pd.read_csv(report_dir / "comparison_folds.csv")
    baseline_folds = report_folds[
        report_folds["Model"] == "Baseline (training mean)"
    ].sort_values("Fold")

    fold_numbers = baseline_folds["Fold"].to_numpy()
    fit_counts = baseline_folds["Fit matches"].to_numpy()

    for name, label, color, marker in [
        ("Baseline (training mean)", "Baseline", "#777777", "s"),
        ("FIFA points only", "FIFA-only", "#B56B25", "^"),
        ("Ridge (8 inputs)", "Ridge", "#345C84", "o"),
    ]:
        model_folds = report_folds[
            report_folds["Model"] == name
        ].set_index("Fold").loc[fold_numbers]

        mean_rmse = report_cv.loc[name, "Mean RMSE"]

        axes[0].plot(
            fold_numbers, model_folds["RMSE"],
            color=color, marker=marker,
            linewidth=1.5, markersize=4,
            label=f"{label} (mean {mean_rmse:.4f})"
        )

    axes[0].set_xticks(fold_numbers, [
        f"Fold {fold}\n{count} fit matches"
        for fold, count in zip(fold_numbers, fit_counts)
    ])
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Validation RMSE (goals)")
    axes[0].set_title(
        "A  RMSE across validation windows",
        fontsize=9.5, loc="left"
    )
    axes[0].margins(x=0.08, y=0.22)
    axes[0].legend(
        loc="upper right", fontsize=7.5, frameon=False
    )
    axes[0].grid(color="#E2E2E2", linewidth=0.6)
    axes[0].set_axisbelow(True)

    # Panel B: Ridge training residuals
    report_residuals = y_train - ridge_model.predict(X_train)
    quantiles, fitted_line = stats.probplot(
        report_residuals, dist="norm"
    )

    axes[1].scatter(
        quantiles[0], quantiles[1],
        s=10, color="#345C84", alpha=0.85
    )

    line_x = np.array([min(quantiles[0]), max(quantiles[0])])
    axes[1].plot(
        line_x,
        fitted_line[0] * line_x + fitted_line[1],
        color="#9E4C3E",
        linewidth=1.3
    )

    axes[1].set_title(
        "B  Residuals have a long upper tail",
        fontsize=9.5, loc="left"
    )
    axes[1].set_xlabel("Theoretical normal quantiles")
    axes[1].set_ylabel("Ordered residuals (goals)")
    axes[1].grid(alpha=0.18)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figure_dir = BASE_DIR / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(w_pad=1.4, pad=0.6)
    report_figure = figure_dir / "05_report_model_comparison.png"
    fig.savefig(report_figure, dpi=320)
    plt.close(fig)

    print("\nSTEP 5: REPORT FIGURE SAVED")
    print(report_figure)


# ADDITIONAL CHECKS: Ridge versus baseline and FIFA-only
if __name__ == "__main__":
    from scipy import stats

    check_rows = []
    ridge_absolute_error = (
        test_predictions["act_goals"] - test_predictions["pred_ridge"]
    ).abs()

    for reference, prediction_column in [
        ("Baseline", "pred_baseline"),
        ("FIFA points only", "pred_fifa_only"),
    ]:
        reference_error = (
            test_predictions["act_goals"]
            - test_predictions[prediction_column]
        ).abs()

        # Average the two teams' error differences within each match.
        gains = (reference_error - ridge_absolute_error).groupby(
            test_predictions["MatchNumber"]
        ).mean()

        t_test = stats.ttest_1samp(gains, popmean=0)
        ci = t_test.confidence_interval(confidence_level=0.95)
        wilcoxon = stats.wilcoxon(gains, alternative="two-sided")

        check_rows.append({
            "Comparison": "Ridge vs " + reference,
            "Matches": len(gains),
            "Mean MAE gain": gains.mean(),
            "95% CI lower": ci.low,
            "95% CI upper": ci.high,
            "t-test p": t_test.pvalue,
            "Wilcoxon p": wilcoxon.pvalue,
        })

    paired_checks = pd.DataFrame(check_rows)
    paired_checks.to_csv(
        test_output_dir / "additional_paired_checks.csv", index=False
    )

    print("\nADDITIONAL PAIRED CHECKS")
    print("Positive gain means lower match-level MAE for Ridge.")
    print(paired_checks.round(4).to_string(index=False))
    print("Exploratory results; matches may not be independent.")
    print("Wilcoxon also assumes symmetric error differences.")