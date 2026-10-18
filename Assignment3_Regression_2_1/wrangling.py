import pandas as pd

df = pd.read_excel("data_wrangling_filled.xlsx", sheet_name="in")

df = df.rename(columns={
    "Shot accurrancy": "shot_accuracy",
    "Pass acurracy": "pass_accuracy",
    "FIFA ranking points": "fifa_ranking_points"
})

features = [
    "fifa_ranking_points",
    "shots_per90",
    "shot_accuracy",
    "goals_conceded_per90",
    "saving",
    "tackles_attempted_per90",
    "fouls_committed_per90",
    "cards_per90"
]

assert len(df) == 208
assert df["MatchNumber"].nunique() == 104
assert df.groupby("MatchNumber").size().eq(2).all()

rows = []

# Keep the workbook's team order: all differences are Team A minus Team B.
for match_id, group in df.groupby("MatchNumber", sort=True):
    a = group.iloc[0]
    b = group.iloc[1]

    row = {
        "MatchNumber": match_id,
        "Date": a["Date"],
        "Round": a["Round"],
        "Team_A": a["Team"],
        "Team_B": b["Team"]
    }

    for col in features:
        row[col + "_diff"] = round(a[col] - b[col], 4)

    # The target is this match's score difference; inputs describe earlier matches.
    row["goal_difference"] = a["act_goals"] - b["act_goals"]
    rows.append(row)

lr21 = pd.DataFrame(rows)

lr21.to_csv("regression_2_1.csv", index=False)

print("Regression 2.1:", lr21.shape)
print(lr21.head())
