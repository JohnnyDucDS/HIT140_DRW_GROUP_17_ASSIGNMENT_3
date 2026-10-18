import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Use the five predictors retained after the development comparisons.
features_5 = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff"
]

target = "goal_difference"

# Keep the original split and CSV row order so the test matches stay the same.
train, test = train_test_split(df, test_size=0.20, random_state=42)

assert len(train) == 83 and len(test) == 21, "Expected 104 matches (83 train + 21 test)."
assert set(train["MatchNumber"]).isdisjoint(set(test["MatchNumber"]))

print("FINAL LINEAR REGRESSION 2.1")
print("Training matches:", len(train), "| Test matches:", len(test))
print("Selected predictors:", len(features_5))

final_model = LinearRegression()
final_model.fit(train[features_5], train[target])

y_pred = final_model.predict(test[features_5])

original_model = LinearRegression()
original_model.fit(train[features_8], train[target])
y_pred_original = original_model.predict(test[features_8])

# The baseline uses the development mean, not the test mean.
baseline = DummyRegressor(strategy="mean")
baseline.fit(train[features_5], train[target])
y_pred_baseline = baseline.predict(test[features_5])

print("\nSELECTED FIVE-VARIABLE REGRESSION EQUATION")
print(f"Intercept: {final_model.intercept_:.6f}")
for feature, coefficient in zip(features_5, final_model.coef_):
    print(f"{feature:32s} {coefficient:+.6f}")

# Fit on the same 83 matches to get coefficient intervals and p-values.
X_inference = sm.add_constant(train[features_5], has_constant="add")
ols_inference = sm.OLS(train[target], X_inference).fit()
confidence_intervals = ols_inference.conf_int(alpha=0.05)
coefficient_table = pd.DataFrame({
    "Coefficient": ols_inference.params,
    "CI lower": confidence_intervals[0],
    "CI upper": confidence_intervals[1],
    "p-value": ols_inference.pvalues
})
print("\nSELECTED MODEL COEFFICIENTS AND 95% CONFIDENCE INTERVALS")
print(coefficient_table.round(6).to_string())

ranking_coef = ols_inference.params["fifa_ranking_points_diff"]
ranking_ci = confidence_intervals.loc["fifa_ranking_points_diff"]
print("\nINTERPRETATION (holding other predictors constant)")
print(f"For +100 FIFA ranking points in Team A minus Team B, "
      f"predicted goal difference changes by {100 * ranking_coef:+.3f} goals "
      f"(95% CI: {100 * ranking_ci.iloc[0]:+.3f} to "
      f"{100 * ranking_ci.iloc[1]:+.3f}).")
print("Note: These are conventional OLS confidence intervals, conditional on "
      "the selected five-variable model; feature selection used the same "
      "development data.")

# Check whether residual variance changes with the predictors.
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
    ols_inference.resid, ols_inference.model.exog
)
print("\nBREUSCH-PAGAN TEST (training residuals)")
print(f"LM statistic: {bp_lm:.4f} | LM p-value: {bp_lm_p:.4f}")
print(f"F statistic: {bp_f:.4f} | F p-value: {bp_f_p:.4f}")

# Check which development matches most influence the fitted coefficients.
cooks_distance = ols_inference.get_influence().cooks_distance[0]
# 4/n is a review flag, not a rule for removing matches.
cook_threshold = 4 / len(train)
max_position = int(np.argmax(cooks_distance))
max_match = train.iloc[max_position]
print("\nCOOK'S DISTANCE (training observations)")
print(f"Maximum Cook's distance: {cooks_distance[max_position]:.4f}")
print(f"Screening guideline (4/n): {cook_threshold:.4f}")
print(f"Observations above guideline: {int((cooks_distance > cook_threshold).sum())}")
print(f"Most influential match: {max_match['MatchNumber']} "
      f"({max_match['Team_A']} vs {max_match['Team_B']})")
print("Note: Exceeding 4/n flags an observation for review, not automatic removal.")

# Training fit is descriptive; use the test scores for final performance.
train_pred = final_model.predict(train[features_5])
train_r2 = r2_score(train[target], train_pred)
n = len(train)
p = len(features_5)
adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
print(f"\nTraining R²: {train_r2:.4f}")
print(f"Training adjusted R²: {adjusted_r2:.4f}")

comparison = []
for name, prediction in [
    ("Selected OLS (5)", y_pred),
    ("Original OLS (8)", y_pred_original),
    ("Training-mean baseline", y_pred_baseline)
]:
    comparison.append({
        "Model": name,
        "MAE": mean_absolute_error(test[target], prediction),
        "RMSE": np.sqrt(mean_squared_error(test[target], prediction)),
        "R2": r2_score(test[target], prediction)
    })

print("\nFINAL TEST RESULTS (21 matches)")
print(pd.DataFrame(comparison).set_index("Model").round(4).to_string())

predictions = test[["MatchNumber", "Team_A", "Team_B", target]].copy()
predictions["Predicted_5"] = y_pred
predictions["Error_5"] = predictions[target] - predictions["Predicted_5"]
predictions = predictions.sort_values("MatchNumber")
print("\nTEST MATCH PREDICTIONS (selected 5-variable model)")
print(predictions.round(3).to_string(index=False))

plt.figure(figsize=(7, 6))

plt.scatter(
    test[target],
    y_pred,
    color="steelblue",
    alpha=0.7,
    s=65
)

min_val = min(test[target].min(), y_pred.min()) - 0.5
max_val = max(test[target].max(), y_pred.max()) + 0.5

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="Perfect prediction"
)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.xlabel("Actual goal difference")
plt.ylabel("Predicted goal difference")
plt.title("Actual vs Predicted Goal Difference (21 Test Matches)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("LR_1_Actual_vs_Predicted.png", dpi=300)
plt.show()
