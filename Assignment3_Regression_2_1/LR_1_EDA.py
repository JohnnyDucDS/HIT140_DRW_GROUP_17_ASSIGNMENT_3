import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# These predictions come from held-out CV folds, not the reserved test set.
predictions = pd.read_csv("cv_predictions.csv")

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

assert len(predictions) == 83, "Expected 83 development matches."

print("\nDESCRIPTIVE STATISTICS (83 DEVELOPMENT MATCHES)")

y = predictions["goal_difference"]

print("Mean:", round(y.mean(), 3))
print("SD:", round(y.std(), 3))
print("Minimum:", y.min())
print("Maximum:", y.max())
print("Draw percentage:", round((y == 0).mean() * 100, 2))

print(f"\n10 BIGGEST PREDICTION ERRORS (OUT OF {len(predictions)} MATCHES)")
id_columns = [c for c in ["MatchNumber", "Team_A", "Team_B"] if c in predictions]
print(predictions.nlargest(10, "absolute_error")[
    id_columns + [target, "predicted_goal_difference", "residual", "absolute_error"]
].round(3).to_string(index=False))

X = predictions[features]
y = predictions[target]
corr = X.corr()

print("\nCORRELATION WITH GOAL DIFFERENCE")
print(predictions[features + [target]].corr()[target]
      .drop(target).sort_values(ascending=False).round(3))

# Check each predictor pair once, leaving out self-correlations.
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairwise = upper.stack()
if not pairwise.empty:
    strongest = pairwise.abs().idxmax()
    r = corr.loc[strongest[0], strongest[1]]
    print("\nStrongest predictor pair:", strongest[0], "and", strongest[1])
    print("Correlation:", round(r, 4))
    if abs(r) < 0.8:
        print("No predictor pair has |correlation| >= 0.8; do not remove a variable automatically.")

fig, ax = plt.subplots(figsize=(11, 8))
heatmap = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)), labels=features, rotation=45, ha="right")
ax.set_yticks(range(len(features)), labels=features)
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(heatmap, ax=ax, label="Pearson correlation")
ax.set_title("Training-set predictor correlation heatmap")
fig.tight_layout()

# OLS and VIF describe the training data; use CV errors to assess predictions.
X_sm = sm.add_constant(X, has_constant="add")
ols = sm.OLS(y, X_sm).fit()
print("\nSTATSMODELS SUMMARY (TRAINING DATA)")
print(ols.summary())

vif = pd.DataFrame({
    "Variable": features,
    "VIF": [variance_inflation_factor(X_sm.values, i + 1)
            for i in range(len(features))]
})
print("\nVARIANCE INFLATION FACTORS")
print(vif.round(3).to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)

for ax, feature in zip(axes.flat, features):
    x_values = predictions[feature].to_numpy(dtype=float)
    y_values = predictions[target].to_numpy(dtype=float)

    ax.scatter(x_values, y_values, alpha=0.55, s=25)

    if np.unique(x_values).size >= 3:
        slope, intercept = np.polyfit(x_values, y_values, deg=1)
        x_grid = np.linspace(x_values.min(), x_values.max(), 100)
        ax.plot(x_grid, intercept + slope * x_grid,
                color="black", linestyle="--", linewidth=1.2)

        # LOWESS helps spot curvature that a straight line can miss.
        lowess = sm.nonparametric.lowess(
            y_values, x_values, frac=0.65, return_sorted=True
        )
        ax.plot(lowess[:, 0], lowess[:, 1],
                color="tab:orange", linewidth=1.7)

    ax.set_title(feature.replace("_diff", "").replace("_", " "), fontsize=10)
    ax.set_xlabel("Team A - Team B", fontsize=9)
    ax.grid(alpha=0.15)

for ax in axes[:, 0]:
    ax.set_ylabel("Actual goal difference", fontsize=9)

fig.suptitle("Predictor vs goal difference (83 training matches)", fontsize=14)
fig.text(0.5, 0.01,
         "Dashed black: simple linear trend | Orange: LOWESS smoother (exploratory only)",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])

fig.savefig("images/Figure_2_EDA.png", dpi=300, bbox_inches="tight")
plt.show()
