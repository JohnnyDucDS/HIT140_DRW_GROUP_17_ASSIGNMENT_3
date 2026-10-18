import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("regression_2_1.csv")

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

# Match the original split so experiments use the same development matches.
train, test = train_test_split(df, test_size=0.20, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compare dropping weak candidates separately and in combinations.
experiments = {
    "Original (8)": [],
    "No fouls (7)": ["fouls_committed_per90_diff"],
    "No cards (7)": ["cards_per90_diff"],
    "No accuracy (7)": ["shot_accuracy_diff"],
    "No cards + accuracy (6)": [
    "cards_per90_diff",
    "shot_accuracy_diff"
    ],
    "No fouls + accuracy (6)": ["fouls_committed_per90_diff", "shot_accuracy_diff"],
    "No fouls + cards (6)": ["fouls_committed_per90_diff", "cards_per90_diff"],
    "No fouls + cards + accuracy (5)": [
        "fouls_committed_per90_diff", "cards_per90_diff", "shot_accuracy_diff"
    ],
    "Signed-log ranking (5)": [
        "fouls_committed_per90_diff", "cards_per90_diff", "shot_accuracy_diff"
    ],
    "Yeo-Johnson ranking (5)": [
        "fouls_committed_per90_diff", "cards_per90_diff", "shot_accuracy_diff"
    ],
    "Yeo-Johnson all (5)": [
        "fouls_committed_per90_diff", "cards_per90_diff", "shot_accuracy_diff"
    ]
}

results = []
for name, removed in experiments.items():
    selected_features = [col for col in features if col not in removed]

    for fold, (fit_idx, val_idx) in enumerate(kf.split(train), start=1):
        fit_data = train.iloc[fit_idx]
        val_data = train.iloc[val_idx]

        X_fit = fit_data[selected_features].copy()
        X_val = val_data[selected_features].copy()

        # Signed log keeps the sign of negative ranking differences.
        if name == "Signed-log ranking (5)":
            ranking = "fifa_ranking_points_diff"
            for X_part in (X_fit, X_val):
                x = X_part[ranking].to_numpy()
                X_part[ranking] = np.sign(x) * np.log1p(np.abs(x))

        # Fit each transformation inside the fold to avoid using validation data.
        if name in ["Yeo-Johnson ranking (5)", "Yeo-Johnson all (5)"]:
            if name == "Yeo-Johnson ranking (5)":
                transform_features = ["fifa_ranking_points_diff"]
            else:
                transform_features = selected_features

            scaler = PowerTransformer()
            std_x_fit = scaler.fit_transform(X_fit[transform_features].values)
            std_x_val = scaler.transform(X_val[transform_features].values)

            std_x_fit_df = pd.DataFrame(
                std_x_fit, index=X_fit.index, columns=transform_features
            )
            std_x_val_df = pd.DataFrame(
                std_x_val, index=X_val.index, columns=transform_features
            )

            X_fit[transform_features] = std_x_fit_df
            X_val[transform_features] = std_x_val_df

        model = LinearRegression()
        model.fit(X_fit, fit_data[target])
        y_pred = model.predict(X_val)

        results.append({
            "Model": name,
            "Fold": fold,
            "MAE": mean_absolute_error(val_data[target], y_pred),
            "RMSE": np.sqrt(mean_squared_error(val_data[target], y_pred)),
            "R2": r2_score(val_data[target], y_pred)
        })

results = pd.DataFrame(results)
summary = results.groupby("Model", sort=False)[["MAE", "RMSE", "R2"]].mean()
summary["MAE change"] = summary["MAE"] - summary.loc["Original (8)", "MAE"]
summary_sorted = summary.sort_values("MAE")

print("Training matches:", len(train), "| Reserved test matches:", len(test))
print("\nAVERAGE FIVE-FOLD RESULTS (sorted by MAE; lower is better)")
print(summary_sorted.round(4).to_string())

for metric in ["MAE", "RMSE", "R2"]:
    table = results.pivot(index="Model", columns="Fold", values=metric)
    table.columns = [f"Fold {i}" for i in table.columns]
    table = table.loc[summary_sorted.index]
    table["Average"] = summary[metric]
    print(f"\n{metric} COMPARISON BY FOLD")
    print(table.round(4).to_string())
