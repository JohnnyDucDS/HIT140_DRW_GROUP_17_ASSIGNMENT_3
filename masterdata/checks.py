"""Consistency checks on collected raw data (run after collect.py, before prepare.py)."""
import pandas as pd

from common import INTERMEDIATE, same_team

raw = pd.read_csv(INTERMEDIATE / "raw_history.csv")
cross = pd.read_csv(INTERMEDIATE / "crosscheck.csv")
lists = pd.read_csv(INTERMEDIATE / "friendly_lists.csv")

print("raw records:", len(raw), raw["match_type"].value_counts().to_dict())
print("WC records per team (expect 3..8):", raw[raw.match_type == "WC2026"].groupby("team").size().describe().to_dict())

# 1. Link coverage
for kind in ["WC2026", "FRIENDLY2026"]:
    sub = raw[raw.match_type == kind]
    print(f"{kind}: fotmob linked {sub.fotmob_id.notna().mean():.0%}, espn linked {sub.espn_id.notna().mean():.0%}")
    print("   unlinked fotmob:", sub[sub.fotmob_id.isna()][["team", "opponent", "kickoff_utc"]].values.tolist())

# 2. Field coverage by match type
fields = ["shots_total", "shots_on_target", "passes_att", "passes_cmp", "possession_pct", "saves",
          "opp_shots_on_target", "fouls_committed", "opp_fouls_committed", "opp_offsides", "tackles", "corners",
          "distance_km", "yellow_cards", "red_cards", "goals_against"]
print(raw.groupby("match_type")[fields].apply(lambda d: d.notna().mean().round(2)).T.to_string())

# 3. Provider agreement where several providers report the same match
for f in ["shots_total", "shots_on_target", "saves", "fouls_committed", "tackles", "corners", "offsides",
          "yellow_cards", "red_cards", "passes_att", "possession_pct"]:
    for other in ["fotmob", "espn"]:
        d = cross[[f"primary_{f}", f"{other}_{f}"]].dropna()
        if len(d):
            diff = (d.iloc[:, 0] - d.iloc[:, 1]).abs()
            print(f"{f:16s} primary vs {other:6s}: n={len(d):3d} exact={(diff < 0.51).mean():.0%} "
                  f"max|diff|={diff.max():.1f}")
d = cross.dropna(subset=["fifa_fdh_yellow", "primary_yellow_cards"])
print("FIFA FDH yellow vs bookings-derived:", f"n={len(d)} exact={(d.fifa_fdh_yellow == d.primary_yellow_cards).mean():.0%}")
d = cross.dropna(subset=["fifa_fdh_red", "primary_red_cards"])
print("FIFA FDH red vs bookings-derived:", f"n={len(d)} exact={(d.fifa_fdh_red == d.primary_red_cards).mean():.0%}")

# 4. Scores: FIFA vs FotMob
bad = []
for r in cross.itertuples():
    if isinstance(r.fotmob_score, str):
        h, a = (int(x) for x in r.fotmob_score.split(" - "))
        home_is_team = isinstance(r.fotmob_home, str) and same_team(r.team, r.fotmob_home)
        mine = h if home_is_team else a
        if mine != r.fifa_goals_for:
            bad.append((r.team, r.opponent, r.kickoff_utc, r.fifa_goals_for, r.fotmob_score, r.fotmob_home))
print("score disagreements FIFA vs FotMob:", bad)

# 5. Extra time flags vs FotMob reason
et = raw[(raw.extra_time) | raw.fotmob_reason.isin(["AET", "Pen", "AP"])]
print("extra-time matches:", et[["team", "opponent", "kickoff_utc", "result_type", "fotmob_reason"]].values.tolist())
print("awarded matches in raw:", raw[raw.awarded == True][["team", "opponent", "kickoff_utc"]].values.tolist())

# 6. FIFA friendly list vs FotMob team fixtures: FIFA's last friendly should be FotMob's newest match before the
#    opener; a newer FotMob match would be a friendly that FIFA omits.
fifa1 = lists[(lists.source == "FIFA teamform (friendlies)") & (lists["rank"] == 1)].set_index("team")
any1 = lists[lists.source == "FIFA teamform (any competition)"].set_index("team")
fm = lists[lists.source == "FotMob team fixtures"]
not_friendly = [t for t in any1.index if any1.loc[t, "fifa_match_id"] != fifa1.loc[t, "fifa_match_id"]]
print("teams whose last match before the opener is not a friendly (FIFA):", not_friendly)
agree, issues = 0, []
for team, x in fifa1.iterrows():
    g = fm[fm.team == team].sort_values("rank")
    if g.empty:
        issues.append((team, "no FotMob fixtures"))
        continue
    t_fifa = pd.Timestamp(x.utc, tz="UTC")
    times = [pd.Timestamp(u, tz="UTC") for u in g.utc]
    if abs((times[0] - t_fifa).total_seconds()) < 36 * 3600:
        agree += 1
    elif times[0] > t_fifa:
        issues.append((team, "FotMob has a NEWER match", g.iloc[0].utc, g.iloc[0].home, g.iloc[0].away))
    else:
        issues.append((team, "FotMob does not list FIFA's last friendly", x.utc, x.home, x.away,
                       "FotMob newest:", g.iloc[0].utc, g.iloc[0].home, g.iloc[0].away))
print(f"FIFA last friendly = FotMob newest match before the opener: {agree}/{len(fifa1)}")
for i in issues:
    print("   ", i)
