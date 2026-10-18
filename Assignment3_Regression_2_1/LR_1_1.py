import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics as metrics

df = pd.read_csv("regression_2_1.csv")

# All inputs are Team A minus Team B.
features = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "shot_accuracy_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff",
    "fouls_committed_per90_diff",
    "cards_per90_diff"
]
target = "goal_difference"

assert len(df) == 104, "Expected 104 matches."
assert df[features + [target]].notna().all().all(), "Missing values detected."

# Keep CSV row order unchanged so later scripts use the same 83/21 split.
train, test = train_test_split(df, test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(train[features], train[target])

y_train_pred = model.predict(train[features])
y_test_pred = model.predict(test[features])
r2_train = r2_score(train[target], y_train_pred)
n = len(train)
p = len(features)
adjusted_r2 = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)

# The baseline uses the training mean, without using test results.
baseline = DummyRegressor(strategy="mean")
baseline.fit(train[features], train[target])

y_test = test[target]
y_pred = model.predict(test[features])
y_pred_base = baseline.predict(test[features])

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Mean Baseline"],
    "MAE": [
        metrics.mean_absolute_error(y_test, y_pred),
        metrics.mean_absolute_error(y_test, y_pred_base)
    ],
    "RMSE": [
        np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        np.sqrt(metrics.mean_squared_error(y_test, y_pred_base))
    ],
    "R2": [
        metrics.r2_score(y_test, y_pred),
        metrics.r2_score(y_test, y_pred_base)
    ]
})

print("INITIAL LINEAR REGRESSION")
print("Training matches:", len(train), "| Testing matches:", len(test))
print("Intercept:", model.intercept_)
print(pd.Series(model.coef_, index=features).round(6).to_string())
print("Training R²:", round(r2_train, 4))
print("Training Adjusted R²:", round(adjusted_r2, 4))
print("Testing MAE:", round(mean_absolute_error(test[target], y_test_pred), 4))
print("Testing RMSE:", round(np.sqrt(mean_squared_error(test[target], y_test_pred)), 4))
print("Testing R²:", round(r2_score(test[target], y_test_pred), 4))

print("\nTEST RESULTS: MODEL VS BASELINE")
print(comparison.round(4).to_string(index=False))

# Cross-validation uses only the 83 development matches.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
prediction_rows = []
metric_rows = []

for fold, (fit_idx, val_idx) in enumerate(kf.split(train), start=1):
    fit_data = train.iloc[fit_idx]
    val_data = train.iloc[val_idx]

    for name, estimator in [
        ("LinearRegression", LinearRegression()),
        ("MeanBaseline", DummyRegressor(strategy="mean"))
    ]:
        estimator.fit(fit_data[features], fit_data[target])
        predictions = estimator.predict(val_data[features])

        metric_rows.append({
            "fold": fold,
            "model": name,
            "training_matches": len(fit_data),
            "validation_matches": len(val_data),
            "MAE": mean_absolute_error(val_data[target], predictions),
            "RMSE": np.sqrt(mean_squared_error(val_data[target], predictions)),
            "R2": r2_score(val_data[target], predictions)
        })

        # Each match gets one prediction from a model fitted without that match.
        if name == "LinearRegression":
            fold_results = val_data.copy()
            fold_results.insert(0, "fold", fold)
            fold_results["predicted_goal_difference"] = predictions
            fold_results["residual"] = (fold_results[target] - predictions)
            fold_results["absolute_error"] = fold_results["residual"].abs()
            prediction_rows.append(fold_results)

cv_predictions = pd.concat(prediction_rows).sort_index()
cv_metrics = pd.DataFrame(metric_rows)
cv_predictions.to_csv("cv_predictions.csv", index=False)
cv_metrics.to_csv("cv_metrics.csv", index=False)

print("\nFIVE-FOLD CROSS-VALIDATION")
print(cv_metrics.round(4).to_string(index=False))
print("\nAVERAGE METRICS")
print(cv_metrics.groupby("model")[["MAE", "RMSE", "R2"]].mean().round(4))
print("\nSaved cv_predictions.csv (83 validation predictions)")
print("Saved cv_metrics.csv (5 folds x 2 models)")
