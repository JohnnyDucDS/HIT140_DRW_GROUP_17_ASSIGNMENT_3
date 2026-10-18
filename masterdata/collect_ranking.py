"""Collect FIFA men's world ranking points for the 48 World Cup 2026 teams.

Output (in data_wrangling/intermediate/):
  fifa_ranking.csv  one row per ranking release x WC team: rank, total points, previous points

Releases are fetched by FIFA's numeric dateId. The FIFA ranking page lists 2026 releases under
'FRS_Male_Football_*' ids, but the ranking API returns an empty list for those, so the numeric ids
were found by scanning the API (id14871..id15179; only these returned 2026 data):
  id15065  released 2026-04-01  (the release before the tournament ranking)
  id15136  released 2026-06-11 10:00 UTC  (9 hours before the opener kicks off at 19:00 UTC)
  id15175  released 2026-07-20 10:00 UTC  (day after the final; its previous points cross-check June)

prepare.py picks, for each row, the latest release published before the row's kickoff.

Usage:
  python3 collect_ranking.py
"""
from collect import write_csv
from common import FIFA_CALENDAR, FIFA_RANKING, RANKING_CSV, fetch_json

RELEASES = ["id15065", "id15136", "id15175"]


def wc_teams():
    """IdTeam -> FIFA name for the 48 teams in the WC 2026 calendar."""
    teams = {}
    for m in fetch_json(FIFA_CALENDAR, "fifa_calendar")["Results"]:
        for side in ("Home", "Away"):
            teams[m[side]["IdTeam"]] = m[side]["TeamName"][0]["Description"]
    return teams


def main():
    teams = wc_teams()
    assert len(teams) == 48, len(teams)
    rows, by_release = [], {}
    for date_id in RELEASES:
        url = FIFA_RANKING.format(date_id=date_id)
        ranking = fetch_json(url, "fifa_ranking")["rankings"]
        assert ranking, f"empty ranking for {date_id}"
        by_id = {x["rankingItem"]["idTeam"]: x for x in ranking}
        missing = [n for i, n in teams.items() if i not in by_id]
        assert not missing, (date_id, missing)
        released = ranking[0]["lastUpdateDate"]
        by_release[date_id] = {}
        for id_team, name in sorted(teams.items(), key=lambda t: t[1]):
            x = by_id[id_team]
            item = x["rankingItem"]
            assert item["name"] == name, (date_id, item["name"], name)
            by_release[date_id][id_team] = item["totalPoints"]
            rows.append({"date_id": date_id, "release_utc": released, "fifa_id_team": id_team, "team": name,
                         "rank": item["rank"], "total_points": item["totalPoints"],
                         "previous_points": x["previousPoints"], "previous_rank": item.get("previousRank"),
                         "confederation": (x.get("tag") or {}).get("id"), "url": url})
        print(f"{date_id}: released {released}, {len(ranking)} teams, 48/48 WC teams found")

    # cross-check: each release's 'previous points' must equal the preceding release's total points
    for row in rows:
        i = RELEASES.index(row["date_id"])
        if i == 0:
            row["previous_release_check"] = None
            continue
        before = by_release[RELEASES[i - 1]][row["fifa_id_team"]]
        row["previous_release_check"] = abs(row["previous_points"] - before) < 0.005
    for date_id in RELEASES[1:]:
        ok = sum(1 for r in rows if r["date_id"] == date_id and r["previous_release_check"])
        print(f"{date_id}: previous points equal the preceding release's total points for {ok}/48 teams")

    write_csv(RANKING_CSV, rows)
    print(f"wrote {RANKING_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
