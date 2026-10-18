"""Create two OLS diagnostic plots and a training correlation heatmap"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Save PNG files directly, without opening blocking plot windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from prepare_data import BASE_DIR, X_train, y_train, X_test


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#2463A6"
DARK = "#27374D"
ORANGE = "#B77818"


def make_scatter_axes(title, subtitle):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.82)
    fig.text(0.14, 0.94, title, fontsize=16, fontweight="semibold", color=DARK)
    fig.text(0.14, 0.885, subtitle, fontsize=10, color="#526174")
    ax.grid(True, color="#E4E9EF", linewidth=0.7)
    ax.tick_params(colors="#526174")
    return fig, ax


def plot_actual_vs_predicted(results, destination):
    actual = results["act_goals"]
    predicted = results["pred_goals"]
    matches = results["MatchNumber"].nunique()
    fig, ax = make_scatter_axes("OLS: actual and predicted goals",
                                f"Test set: {len(results)} team-match rows from {matches} matches",)
    ax.scatter(actual, predicted, s=53, color=BLUE, alpha=0.80,
               edgecolors="white", linewidths=0.7, zorder=3,)

    # Equal axis scales make the perfect-prediction line interpretable.
    low = float(np.floor(min(0, actual.min(), predicted.min()))) - 0.25
    high = float(np.ceil(max(actual.max(), predicted.max()))) + 0.25
    ax.plot([low, high], [low, high], linestyle="--", color=DARK,
            linewidth=1.3, label="Perfect prediction", zorder=2,)
    baseline = results["baseline_goals"].iloc[0]
    ax.axhline(baseline, linestyle=":", color=ORANGE, linewidth=1.6,
               label=f"Training-mean baseline ({baseline:.2f})", zorder=2,)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Actual goals per team-match")
    ax.set_ylabel("Predicted goals per team-match")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(results, destination):
    predicted = results["pred_goals"]
    residual = results["residual"]
    fig, ax = make_scatter_axes("OLS: residuals and predicted goals",
                                "Residual = actual goals - predicted goals. Test set: "
                                f"{len(results)} team-match rows.",)
    ax.scatter(predicted, residual, s=53, color=BLUE, alpha=0.80,
              edgecolors="white", linewidths=0.7, zorder=3,)
    ax.axhline(0, color=DARK, linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicted goals per team-match")
    ax.set_ylabel("Residual (goals)")
    ax.margins(x=0.08, y=0.12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_training_correlations(destination):
    # All correlations are calculated from training observations only.
    training_data = X_train.copy()
    training_data["act_goals"] = y_train
    correlations = training_data.corr(method="pearson")
    labels = {
        "fifa_ranking_points": "Team FIFA points",
        "opponent_fifa_ranking_points": "Opp. FIFA points",
        "shots_per90": "Shots per 90",
        "possession": "Possession",
        "corners_per90": "Corners per 90",
        "opponent_goals_conceded_per90": "Opp. conceded per 90",
        "opponent_pass_accuracy": "Opp. pass accuracy",
        "pass_accuracy": "Pass accuracy",
        "act_goals": "Actual goals",
    }
    tick_labels = [labels[name] for name in correlations.columns]
    fig, ax = plt.subplots(figsize=(11.0, 9.5), constrained_layout=True)
    matrix = ax.imshow(
        correlations.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1
    )
    positions = np.arange(len(tick_labels))
    ax.set_xticks(positions, labels=tick_labels, rotation=45, ha="right")
    ax.set_yticks(positions, labels=tick_labels)
    ax.tick_params(axis="both", length=0, labelsize=10)
    ax.set_title(
        "Correlations among predictors and goals\n"
        f"Training set: {len(X_train)} team-match rows. Opp. = opponent.",
        loc="left", fontsize=14, pad=18, color=DARK,
    )
    for row in range(len(tick_labels)):
        for column in range(len(tick_labels)):
            value = correlations.iloc[row, column]
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(value) > 0.60 else DARK,
            )
    boundaries = np.arange(-0.5, len(tick_labels), 1)
    ax.set_xticks(boundaries, minor=True)
    ax.set_yticks(boundaries, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(matrix, ax=ax, shrink=0.78, pad=0.03)
    colorbar.set_label("Pearson correlation")
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = BASE_DIR / "outputs"
    predictions_path = output_dir / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Run regression_2_2.py first to create outputs/test_predictions.csv."
        )
    results = pd.read_csv(predictions_path)
    required = {
        "MatchNumber", "Team", "act_goals", "pred_goals",
        "baseline_goals", "residual",
    }
    if not required.issubset(results.columns):
        raise ValueError("The saved prediction file is missing required columns.")
    if len(results) != len(X_test):
        raise ValueError("Saved predictions do not match the current test-set size.")
    numeric = results[["act_goals", "pred_goals", "baseline_goals", "residual"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Saved predictions contain missing or non-finite values.")
    if not np.allclose(
        results["residual"], results["act_goals"] - results["pred_goals"]
    ):
        raise ValueError("Saved residuals do not equal actual minus predicted goals.")

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_actual_vs_predicted(
        results, figure_dir / "01_actual_vs_predicted.png"
    )
    plot_residuals(
        results, figure_dir / "02_residuals_vs_predicted.png"
    )
    plot_training_correlations(
        figure_dir / "03_training_correlations.png"
    )

    print("PLOTS CREATED")
    print("Test observations:", len(results))
    print("Training observations for correlations:", len(X_train))
    print("Figure folder:", figure_dir)
    print("01_actual_vs_predicted.png")
    print("02_residuals_vs_predicted.png")
    print("03_training_correlations.png")


if __name__ == "__main__":
    main()
