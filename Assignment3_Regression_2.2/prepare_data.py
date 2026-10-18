"""Prepare the selected dataset for Linear Regression 2.2."""

from pathlib import Path
import pandas as pd


# 1. Read the CSV from the same folder as this script.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "regression_2_2_restart.csv"
df = pd.read_csv(DATA_PATH)

# 2. Define exactly eight predictors and the target.
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
required_columns = id_columns + features + ["act_goals"]

# 3. Check the dataset before creating training and test sets.
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")
if df.shape != (208, 13):
    raise ValueError(f"Expected 208 rows and 13 columns; found {df.shape}.")
if df.isna().any().any():
    raise ValueError("The dataset contains missing values.")

date_text = df["Date"].astype(str).str.strip()

iso_dates = pd.to_datetime(
    date_text, format="%Y-%m-%d", errors="coerce"
)
dmy_dates = pd.to_datetime(
    date_text, format="%d/%m/%Y", errors="coerce"
)

df["Date"] = iso_dates.fillna(dmy_dates)

if df["Date"].isna().any():
    raise ValueError("Some dates could not be read. Check the CSV.")
df[features + ["act_goals"]] = df[features + ["act_goals"]].apply(
    pd.to_numeric, errors="raise"
)

match_groups = df.groupby("MatchNumber")
if not (
    df["MatchNumber"].nunique() == 104
    and match_groups.size().eq(2).all()
    and match_groups["Team"].nunique().eq(2).all()
    and match_groups["Date"].nunique().eq(1).all()
):
    raise ValueError("Each of the 104 matches must have two different teams on one date.")

X = df[features].copy()
y = df["act_goals"].copy()

# 4. Split approximately 80/20 in date order.
# Both teams and all matches on the same date stay together.
matches = (
    df[["MatchNumber", "Date"]]
    .drop_duplicates()
    .sort_values(["Date", "MatchNumber"])
)
split_position = int(len(matches) * 0.8)
cutoff_date = matches.iloc[split_position]["Date"]

train_mask = df["Date"] < cutoff_date
test_mask = df["Date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()
y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

train_matches = set(df.loc[train_mask, "MatchNumber"])
test_matches = set(df.loc[test_mask, "MatchNumber"])
if train_matches & test_matches:
    raise ValueError("A match appears in both training and test sets.")

# 5. Print only when this file is run directly.
# Importing these variables from the model script will not repeat the output.
if __name__ == "__main__":
    print("DATA PREPARATION")
    print("Dataset shape:", df.shape)
    print("Matches:", df["MatchNumber"].nunique())
    print("Teams:", df["Team"].nunique())
    print("Missing values:", df.isna().sum().sum())
    print("Predictors:", len(features))
    print("Target:", y.name)
    print()
    print("TRAIN/TEST SPLIT")
    print("Training matches:", len(train_matches))
    print("Test matches:", len(test_matches))
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Matches in both sets:", len(train_matches & test_matches))
    print("Last training date:", df.loc[train_mask, "Date"].max().date())
    print("First test date:", df.loc[test_mask, "Date"].min().date())
    print("\nDESCRIPTIVE STATISTICS: TRAINING GOALS")
    print(f"Range: {y_train.min():.0f} to {y_train.max():.0f}")
    print(f"Mean: {y_train.mean():.2f}")
    print(f"Standard deviation: {y_train.std(ddof=1):.2f}")
    print(f"Zero-goal rows: {100 * y_train.eq(0).mean():.1f}%")
