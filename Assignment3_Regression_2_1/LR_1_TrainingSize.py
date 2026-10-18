import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.linear_model import LinearRegression

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

features_5 = [
    "fifa_ranking_points_diff",
    "shots_per90_diff",
    "goals_conceded_per90_diff",
    "saving_diff",
    "tackles_attempted_per90_diff"
]

# Leave the 21 test matches out of the learning curves.
train, test = train_test_split(df, test_size=0.20, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stop at 66 because that is the smallest fitting fold in five-fold CV.
sizes = [20, 30, 40, 50, 60, 66]

# Signed log preserves the direction of the ranking difference.
X_signed_log = train[features_5].copy()
ranking = "fifa_ranking_points_diff"
x = X_signed_log[ranking].to_numpy()
X_signed_log[ranking] = np.sign(x) * np.log1p(np.abs(x))

experiments = {
    "Original (8)": train[features_8],
    "Reduced (5)": train[features_5],
    "Signed-log ranking (5)": X_signed_log
}

results = []
for name, X in experiments.items():
    actual_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(),
        X,
        train["goal_difference"],
        train_sizes=sizes,
        cv=kf,
        scoring="neg_mean_absolute_error",
        shuffle=True,
        random_state=42
    )

    # sklearn returns negative MAE scores; change the sign for reporting.
    for size, train_row, val_row in zip(actual_sizes, train_scores, val_scores):
        results.append({
            "Model": name,
            "Training matches": int(size),
            "Training MAE": -train_row.mean(),
            "Validation MAE": -val_row.mean(),
            "Validation SD": val_row.std()
        })

results = pd.DataFrame(results)
print("Development matches:", len(train), "| Reserved test matches:", len(test))
print("The 21 reserved test matches are NOT used in this experiment.")
print("\nLEARNING CURVE — FIVE-FOLD VALIDATION")
print(results.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
for name, group in results.groupby("Model", sort=False):
    ax.plot(group["Training matches"], group["Validation MAE"], marker="o", label=name)
ax.set_title("Learning curve: validation MAE vs training size")
ax.set_xlabel("Matches used to fit each fold's regression")
ax.set_ylabel("Mean validation MAE (lower is better)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

chosen = results[results["Model"] == "Reduced (5)"]
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(chosen["Training matches"], chosen["Training MAE"], marker="o", label="Training MAE")
ax2.plot(chosen["Training matches"], chosen["Validation MAE"], marker="o", label="Validation MAE")
ax2.set_title("Reduced model: training vs validation learning curve")
ax2.set_xlabel("Matches used to fit each fold's regression")
ax2.set_ylabel("MAE (lower is better)")
ax2.legend()
ax2.grid(alpha=0.25)
fig2.tight_layout()

plt.show()
