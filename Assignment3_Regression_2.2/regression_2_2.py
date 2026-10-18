"""Train, evaluate and save the selected Linear Regression 2.2 model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from prepare_data import (
    BASE_DIR, df, features, id_columns,
    X_train, X_test, y_train, y_test,
)


def evaluate(actual, predicted):
    """Keep predictions unrounded; normalise RMSE by the actual goal range."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    goal_range = actual.max() - actual.min()
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE (%)": 100 * rmse / goal_range if goal_range > 0 else np.nan,
        "R2": r2_score(actual, predicted),
    }


def check_time_validation():
    """Check the fixed feature set using only the original training data."""
    train_dates = df.loc[X_train.index, "Date"]
    dates = np.array(sorted(train_dates.unique()))
    splitter = TimeSeriesSplit(n_splits=3)
    rows = []

    # Split unique dates, then assign both teams from each match together.
    for fold, (fit_positions, validation_positions) in enumerate(
        splitter.split(dates), start=1
    ):
        fit_mask = train_dates.isin(dates[fit_positions])
        validation_mask = train_dates.isin(dates[validation_positions])
        X_fit = X_train.loc[fit_mask]
        y_fit = y_train.loc[fit_mask]
        X_validation = X_train.loc[validation_mask]
        y_validation = y_train.loc[validation_mask]

        fold_model = LinearRegression().fit(X_fit, y_fit)
        predictions = fold_model.predict(X_validation)
        baseline = np.full(len(y_validation), y_fit.mean())
        model_scores = evaluate(y_validation, predictions)
        baseline_scores = evaluate(y_validation, baseline)

        rows.append({
            "Fold": fold,
            "Fit matches": df.loc[X_fit.index, "MatchNumber"].nunique(),
            "Validation matches": df.loc[
                X_validation.index, "MatchNumber"
            ].nunique(),
            "Last fit date": df.loc[X_fit.index, "Date"].max().date(),
            "First validation date": df.loc[
                X_validation.index, "Date"
            ].min().date(),
            "Model MAE": model_scores["MAE"],
            "Model RMSE": model_scores["RMSE"],
            "Baseline MAE": baseline_scores["MAE"],
            "Baseline RMSE": baseline_scores["RMSE"],
        })

    return pd.DataFrame(rows)


def calculate_vif():
    """Calculate VIF from training predictors, using auxiliary regressions."""
    standard_deviations = X_train.std(ddof=0)
    if standard_deviations.eq(0).any():
        raise ValueError("VIF cannot be calculated for a constant predictor.")

    # Scaling is used only for this diagnostic, not the prediction model.
    scaled = (X_train - X_train.mean()) / standard_deviations
    rows = []
    for variable in features:
        other_variables = scaled.drop(columns=variable)
        auxiliary = LinearRegression().fit(other_variables, scaled[variable])
        r_squared = auxiliary.score(other_variables, scaled[variable])
        vif = np.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        rows.append({"Variable": variable, "VIF": vif})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def main():
    # 1. Check the selected feature set within the training data.
    validation_results = check_time_validation()
    vif_results = calculate_vif()

    # 2. Fit all eight predictors using all 162 training rows (81 matches).
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)

    # 3. Evaluate on the original 46 test rows (23 matches).
    # The constant baseline is calculated from training goals only.
    test_predictions = model.predict(X_test)
    baseline_mean = y_train.mean()
    baseline_predictions = np.full(len(y_test), baseline_mean)
    metrics = pd.DataFrame({
        "Linear regression": evaluate(y_test, test_predictions),
        "Baseline": evaluate(y_test, baseline_predictions),
    }).T

    coefficients = pd.DataFrame({
        "Variable": ["Intercept"] + features,
        "Coefficient": [model.intercept_] + list(model.coef_),
    })
    equation = f"Predicted goals = {model.intercept_:.6f}"
    for name, coefficient in zip(features, model.coef_):
        equation += f" {coefficient:+.6f} * {name}"

    # 4. Save full-precision results for plotting and the report.
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    test_results = df.loc[X_test.index, id_columns + ["act_goals"]].copy()
    test_results["pred_goals"] = test_predictions
    test_results["baseline_goals"] = baseline_predictions
    test_results["residual"] = (
        test_results["act_goals"] - test_results["pred_goals"]
    )

    test_results.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index_label="Model")
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    (output_dir / "regression_equation.txt").write_text(
        equation, encoding="utf-8"
    )
    validation_results.to_csv(output_dir / "time_validation.csv", index=False)
    vif_results.to_csv(output_dir / "training_vif.csv", index=False)

    # 5. Print the main checks and results.
    print("LINEAR REGRESSION 2.2")
    print("Predictors:", len(features))
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    print("\nTRAINING CHECKS")
    print(f"Training R2: {r2_score(y_train, train_predictions):.4f}")
    print(f"Training MAE: {mean_absolute_error(y_train, train_predictions):.4f}")
    print(f"Maximum VIF: {vif_results['VIF'].max():.4f}")

    print("\nTIME VALIDATION: training data only")
    print(validation_results[[
        "Fold", "Fit matches", "Validation matches", "Model RMSE", "Baseline RMSE"
    ]].round(4).to_string(index=False))
    print(f"Mean model RMSE: {validation_results['Model RMSE'].mean():.4f}")
    print(f"Mean baseline RMSE: {validation_results['Baseline RMSE'].mean():.4f}")

    print("\nTEST PERFORMANCE")
    print(f"Baseline prediction: {baseline_mean:.4f} goals")
    print(metrics.round(4).to_string())

    print("\nCOEFFICIENTS")
    print(coefficients.round(6).to_string(index=False))

    print("\nRESULTS SAVED")
    print("Output folder:", output_dir)
    print("Test prediction rows:", len(test_results))


if __name__ == "__main__":
    main()
