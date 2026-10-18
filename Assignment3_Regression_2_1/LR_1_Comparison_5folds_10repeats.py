import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("regression_2_1.csv")

features_8 = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "shot_accuracy_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff",
    "fouls_committed_per90_diff",
    "cards_per90_diff"
]

# Keep the reduced model consistent with the optimisation script.
features_5 = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff"
]

target = "goal_difference"
ranking = "fifa_ranking_points_diff"

# Keep the reserved 21 matches out of the CV comparisons.
train, test = train_test_split(df, test_size=0.20, random_state=42)

# Give every model the same 50 validation folds.
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
model_names = [
    "Original (8)",
    "Reduced (5)",
    "Signed-log ranking (5)",
    "Yeo-Johnson ranking (5)",
    "Mean baseline"
]

results = []
for fold_index, (fit_idx, val_idx) in enumerate(rkf.split(train)):
    repeat = fold_index // 5 + 1
    fold = fold_index % 5 + 1
    fit_data = train.iloc[fit_idx]
    val_data = train.iloc[val_idx]

    for name in model_names:
        if name == "Mean baseline":
            # Use only this fold's fitting mean for the baseline.
            predictions = np.full(len(val_data), fit_data[target].mean())
        else:
            selected = features_8 if name == "Original (8)" else features_5
            X_fit = fit_data[selected].copy()
            X_val = val_data[selected].copy()

            if name == "Signed-log ranking (5)":
                for X_part in (X_fit, X_val):
                    x = X_part[ranking].to_numpy()
                    X_part[ranking] = np.sign(x) * np.log1p(np.abs(x))

            elif name == "Yeo-Johnson ranking (5)":
                # Learn the transformation from this fold's fitting rows.
                transformer = PowerTransformer()
                X_fit[ranking] = transformer.fit_transform(X_fit[[ranking]]).ravel()
                X_val[ranking] = transformer.transform(X_val[[ranking]]).ravel()

            model = LinearRegression()
            model.fit(X_fit, fit_data[target])
            predictions = model.predict(X_val)

        results.append({
            "Repeat": repeat,
            "Fold": fold,
            "Model": name,
            "MAE": mean_absolute_error(val_data[target], predictions),
            "RMSE": np.sqrt(mean_squared_error(val_data[target], predictions)),
            "R2": r2_score(val_data[target], predictions)
        })

results = pd.DataFrame(results)

# Average within each repeat before measuring variation across repeats.
per_repeat = results.groupby(["Model", "Repeat"], as_index=False)[
    ["MAE", "RMSE", "R2"]
].mean()

summary = per_repeat.groupby("Model")[["MAE", "RMSE", "R2"]].agg(
    ["mean", "std"]
)
summary.columns = ["_".join(col) for col in summary.columns]
summary = summary.loc[model_names]
summary["MAE change vs original"] = (
    summary["MAE_mean"] - summary.loc["Original (8)", "MAE_mean"]
)
summary = summary.sort_values("MAE_mean")

print("Development matches:", len(train), "| Reserved test matches:", len(test))
print("Validation: 5 folds x 10 repeats = 50 folds per model")
print("No reserved test match was used to select or fit a model here.")
print("\nREPEATED CROSS-VALIDATION SUMMARY (sorted by mean MAE)")
print(summary.round(4).to_string())
print("\nSD means the variability of the 10 per-repeat mean scores.")

mae_by_fold = results.pivot(index=["Repeat", "Fold"], columns="Model", values="MAE")
mae_by_repeat = per_repeat.pivot(index="Repeat", columns="Model", values="MAE")

model_pairs = [
    ("Original (8)", "Mean baseline"),
    ("Reduced (5)", "Mean baseline"),
    ("Reduced (5)", "Original (8)"),
    ("Signed-log ranking (5)", "Reduced (5)"),
    ("Yeo-Johnson ranking (5)", "Reduced (5)")
]

print("\nCORRECTED PAIRED MODEL COMPARISONS")

for model_a, model_b in model_pairs:
    differences = mae_by_fold[model_a] - mae_by_fold[model_b]

    n = len(differences)
    mean_diff = differences.mean()

    # Approximate correction for overlapping folds: validation/training is about 1/4.
    correction = 1 / n + 1 / 4
    corrected_se = differences.std(ddof=1) * np.sqrt(correction)

    critical_value = t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - critical_value * corrected_se
    ci_upper = mean_diff + critical_value * corrected_se

    # Treat this test as exploratory because the folds overlap.
    t_stat = mean_diff / corrected_se
    p_value = 2 * t.sf(abs(t_stat), df=n - 1)

    print(f"\n{model_a} vs {model_b}")
    print(f"Mean MAE difference: {mean_diff:+.4f}")
    print(f"Approximate 95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"Corrected p-value: {p_value:.4f}")

comparison = []
for name in model_names:
    comparison.append({
        "Model": name,
        "Fold wins vs original (of 50)": int(
            (mae_by_fold[name] < mae_by_fold["Original (8)"]).sum()
        ) if name != "Original (8)" else 0,
        "Repeat wins vs original (of 10)": int(
            (mae_by_repeat[name] < mae_by_repeat["Original (8)"]).sum()
        ) if name != "Original (8)" else 0,
        "Repeat wins vs reduced (of 10)": int(
            (mae_by_repeat[name] < mae_by_repeat["Reduced (5)"]).sum()
        ) if name != "Reduced (5)" else 0,
        "Fold wins vs baseline (of 50)": int(
            (mae_by_fold[name] < mae_by_fold["Mean baseline"]).sum()
        ) if name != "Mean baseline" else 0
    })

print("\nCONSISTENCY COMPARISON (lower MAE wins on the same validation folds)")
print(pd.DataFrame(comparison).set_index("Model").to_string())

print("\nMAE BY REPEAT (mean across the five folds in each repeat)")
print(mae_by_repeat[model_names].round(4).to_string())

print("\nReminder: folds overlap, so 50 scores are not 50 independent tests.")
print("These results compare modelling approaches, not final test performance.")
print("Corrected paired tests are approximate and exploratory; folds overlap.")

plot_order = [
    "Mean baseline",
    "Original (8)",
    "Reduced (5)",
    "Signed-log ranking (5)",
    "Yeo-Johnson ranking (5)"
]

cv_mae = summary.loc[plot_order, "MAE_mean"]
# Error bars show SD across repeat means, not a confidence interval.
cv_sd = summary.loc[plot_order, "MAE_std"]

# Fit on all 83 development matches for the reserved-test plot.
final_model = LinearRegression()
final_model.fit(train[features_5], train[target])

test_predictions = final_model.predict(test[features_5])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Orange marks the selected five-predictor OLS model.
colors = [
    "steelblue",
    "steelblue",
    "darkorange",
    "steelblue",
    "steelblue"
]

axes[0].barh(
    plot_order,
    cv_mae,
    xerr=cv_sd,
    capsize=4,
    color=colors
)

axes[0].invert_yaxis()
axes[0].set_xlabel("Mean CV MAE (goals; lower is better)")
axes[0].set_title("A. Model comparison")
axes[0].grid(axis="x", alpha=0.2)

actual = test[target]

axes[1].scatter(actual, test_predictions, alpha=0.7)

limits = [
    min(actual.min(), test_predictions.min()) - 0.5,
    max(actual.max(), test_predictions.max()) + 0.5
]

axes[1].plot(limits, limits, "k--")
axes[1].set_xlim(limits)
axes[1].set_ylim(limits)

axes[1].set_xlabel("Actual goal difference")
axes[1].set_ylabel("Predicted goal difference")
axes[1].set_title("B. Final model predictions (21 test matches)")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("images\Figure_3_TwoPanel.png", dpi=300, bbox_inches="tight")
plt.show()
