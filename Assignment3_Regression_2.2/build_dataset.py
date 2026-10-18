"""Create the Regression 2.2 CSV from the group's prepared Excel file."""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "data_wrangling_filled (1).xlsx"
CSV_PATH = BASE_DIR / "regression_2_2_restart.csv"

features = [
    "fifa_ranking_points",
    "opponent_fifa_ranking_points",
    "shots_per90",
    "possession",
    "corners_per90",
    "opponent_goals_conceded_per90",
    "opponent_pass_accuracy",
    "pass_accuracy",
]
id_columns = ["MatchNumber", "Date", "Round", "Team"]


def main():
    # 1. Read the selected columns from the prepared workbook.
    source_columns = [
        "MatchNumber", "Date", "Round", "Team",
        "FIFA ranking points", "shots_per90", "possession",
        "corners_per90", "goals_conceded_per90", "Pass acurracy",
        "act_goals",
    ]
    data = pd.read_excel(
        EXCEL_PATH, sheet_name="in", engine="openpyxl",
        usecols=source_columns,
    ).rename(columns={
        "FIFA ranking points": "fifa_ranking_points",
        "Pass acurracy": "pass_accuracy",
    })
    data["Date"] = pd.to_datetime(
        data["Date"], dayfirst=True, errors="raise"
    )

    # 2. Check that each match has two different teams on one date.
    groups = data.groupby("MatchNumber")
    if not (
        len(data) == 208
        and data["MatchNumber"].nunique() == 104
        and groups.size().eq(2).all()
        and groups["Team"].nunique().eq(2).all()
        and groups["Date"].nunique().eq(1).all()
    ):
        raise ValueError("Expected 104 matches with two different teams each.")
    if data.isna().any().any():
        raise ValueError("The selected Excel columns contain missing values.")

    # 3. Join the other team's PRE-MATCH inputs within the same fixture.
    opponents = data[[
        "MatchNumber", "Team", "fifa_ranking_points",
        "pass_accuracy", "goals_conceded_per90",
    ]].rename(columns={
        "Team": "Opponent",
        "fifa_ranking_points": "opponent_fifa_ranking_points",
        "pass_accuracy": "opponent_pass_accuracy",
        "goals_conceded_per90": "opponent_goals_conceded_per90",
    })
    dataset = data.merge(opponents, on="MatchNumber", how="inner")
    dataset = dataset.loc[dataset["Team"] != dataset["Opponent"]].copy()

    # Per-90 values are already scaled in Excel; keep those values.
    dataset = dataset[id_columns + features + ["act_goals"]]
    dataset = dataset.reset_index(drop=True)
    dataset[features + ["act_goals"]] = dataset[
        features + ["act_goals"]
    ].apply(pd.to_numeric, errors="raise")
    if dataset.shape != (208, 13) or dataset.isna().any().any():
        raise ValueError("The final dataset must have 208 complete rows.")

    # 4. Export dates in one consistent format.
    dataset["Date"] = dataset["Date"].dt.strftime("%Y-%m-%d")
    dataset.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("DATASET CREATED")
    print("Dataset shape:", dataset.shape)
    print("Matches:", dataset["MatchNumber"].nunique())
    print("Teams:", dataset["Team"].nunique())
    print("Predictors:", len(features))
    print("Missing values:", dataset.isna().sum().sum())
    print("Saved to:", CSV_PATH)


if __name__ == "__main__":
    main()
