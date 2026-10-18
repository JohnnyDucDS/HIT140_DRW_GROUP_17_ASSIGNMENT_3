import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("regression_2_1.csv")

features = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff"
]
target = "goal_difference"

# Use the same development/test split as the OLS scripts.
train, test = train_test_split(df, test_size=0.20, random_state=42)
assert len(train) == 83 and len(test) == 21
assert not train[features + [target]].isna().any().any(), "Missing values in training data"

# Compare all three algorithms on identical outer folds.
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
results = []

for index, (fit_idx, val_idx) in enumerate(rkf.split(train)):
    fit_data = train.iloc[fit_idx]
    val_data = train.iloc[val_idx]

    # Keep scaling inside the pipeline so validation rows do not affect it.
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    # Tune alpha using only the current outer fitting fold.
    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid={"ridge__alpha": np.logspace(-3, 3, 13)},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42)
    )

    models = {
        "Linear regression (5)": LinearRegression(),
        "Ridge regression (5)": ridge_search,
        "Random forest (5)": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    for name, model in models.items():
        model.fit(fit_data[features], fit_data[target])
        prediction = model.predict(val_data[features])

        row = {
            "Repeat": index // 5 + 1,
            "Fold": index % 5 + 1,
            "Model": name,
            "MAE": mean_absolute_error(val_data[target], prediction),
            "RMSE": np.sqrt(mean_squared_error(val_data[target], prediction)),
            "R2": r2_score(val_data[target], prediction)
        }
        if name == "Ridge regression (5)":
            row["Ridge alpha"] = model.best_params_["ridge__alpha"]
        results.append(row)

results = pd.DataFrame(results)

# Measure variation across repeat means, rather than 50 overlapping folds.
per_repeat = results.groupby(["Model", "Repeat"], as_index=False)[
    ["MAE", "RMSE", "R2"]
].mean()
summary = per_repeat.groupby("Model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
summary.columns = ["_".join(col) for col in summary.columns]

print("Development matches:", len(train), "| Reserved test matches:", len(test))
print("5-fold cross-validation x 10 repeats = 50 outer validation folds per model")
print("All three models use identical folds and the same five explanatory variables.")
print("Ridge alpha is tuned by 3-fold CV within each outer training fold.")
print("The reserved 21 test matches are not used in this comparison.")
print("\nALGORITHM COMPARISON (sorted by mean MAE; lower is better)")
print(summary.sort_values("MAE_mean").round(4).to_string())

mae_by_fold = results.pivot(index=["Repeat", "Fold"], columns="Model", values="MAE")
mae_by_repeat = per_repeat.pivot(index="Repeat", columns="Model", values="MAE")
ols = "Linear regression (5)"
ridge = "Ridge regression (5)"
rf = "Random forest (5)"

print("\nWINS VS LINEAR REGRESSION/TOTAL (LOWER MAE)")
for name in [ridge, rf]:
    print(name)
    print("Across 50 folds:", int((mae_by_fold[name] < mae_by_fold[ols]).sum()), "/ 50")
    print("Across 10 repeats:", int((mae_by_repeat[name] < mae_by_repeat[ols]).sum()), "/ 10")

print("\nMEAN MAE BY REPEAT")
print(mae_by_repeat[[ols, ridge, rf]].round(4).to_string())

ridge_alphas = results.loc[results["Model"] == ridge, "Ridge alpha"]
print("\nRIDGE ALPHA CHOSEN WITHIN EACH TRAINING FOLD")
print("Median:", round(ridge_alphas.median(), 4))
print("Range:", round(ridge_alphas.min(), 4), "to", round(ridge_alphas.max(), 4))

# Feature importance is descriptive and uses development data only.
forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(train[features], train[target])
importances = pd.Series(forest.feature_importances_, index=features)
print("\nRANDOM FOREST FEATURE IMPORTANCE (TRAINING DATA ONLY)")
print(importances.sort_values(ascending=False).round(4).to_string())
