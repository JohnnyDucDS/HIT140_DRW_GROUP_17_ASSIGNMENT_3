"""Collect raw per-match team statistics for the World Cup 2026 prep workbook.

Friendly-only approach (branch data-friendly-approach, user decision 2026-10-16): every team's opener uses
an international friendly, so qualifiers are no longer collected.

Outputs (in data_wrangling/intermediate/):
  raw_history.csv     one row per team per match: WC 2026 matches (both teams) and, for each team, its last
                      international friendly before the opener. If that friendly has no published team
                      statistics, older friendlies are collected until one has them (no age limit).
                      Primary values with source and URL per field.
  crosscheck.csv      the same matches with every provider's value side by side
  friendly_lists.csv  each team's newest friendlies from FIFA, plus the newest matches before the opener in
                      FotMob's team fixtures (to catch a friendly FIFA omits)

Usage:
  python3 collect.py                         # all 48 teams
  python3 collect.py --teams "Mexico,South Africa"   # pilot
"""
import argparse
import csv
import datetime as dt
import re

import openpyxl

from common import (ESPN_PAGE, ESPN_SCOREBOARD, ESPN_SUMMARY, FDH_TEAMS, FIFA_CALENDAR, FIFA_LIVE,
                    FIFA_MATCH_PAGE, FIFA_TEAMFORM, FOTMOB_DAY, FOTMOB_MATCH, FOTMOB_PAGE, INTERMEDIATE, RAW_CSV,
                    WB_TO_FIFA, WORKBOOK_IN, fetch_json, nominal_minutes, same_team)

FRIENDLY_COMP = "cesdwwnxbc5fmajgroc0hqzy2"   # FIFA competition id: international friendlies
N_FR_LIST = 5                          # friendlies per team written to friendly_lists.csv
# Fields that show whether a match has published team statistics at all (cards and score always exist).
STATS_PRESENCE = ["shots_total", "possession_pct", "fouls_committed", "passes_att"]
ESPN_SLUGS = ["fifa.world", "fifa.friendly"]
FOTMOB_TEAM = "https://www.fotmob.com/api/data/teams?id={id}&ccode3=AUS"


def parse_utc(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def tname(side):
    return side["TeamName"][0]["Description"]


# ---------------------------------------------------------------- workbook + FIFA calendar
def load_workbook_rows():
    ws = openpyxl.load_workbook(WORKBOOK_IN)["in"]
    rows = []
    for r in ws.iter_rows(min_row=2, max_row=ws.max_row):
        a, rnd, num, team = (c.value for c in r[:4])
        rows.append({"excel_row": r[0].row, "date_raw": a, "round": rnd, "match_number": num, "team": team})
    return rows


def load_calendar():
    cal = fetch_json(FIFA_CALENDAR, "fifa_calendar")
    by_num = {}
    for m in cal["Results"]:
        by_num[m["MatchNumber"]] = m
    return by_num


# ---------------------------------------------------------------- FIFA per-match data
def fifa_live(id_match):
    return fetch_json(FIFA_LIVE.format(id=id_match), "fifa_live")


def fifa_bookings(live, side):
    """Return (yellow, red, n_events, status). Codes: 1 yellow, 2 direct red, 3 second yellow -> red.

    FIFA logs a second-yellow dismissal as a code-1 yellow plus a code-3 card at the same minute.
    Following the usual convention (Opta/FotMob/ESPN), that dismissal counts as one red and the
    second yellow is not counted again, so the player contributes 1 yellow + 1 red.
    Only player bookings count. Empty booking list with a known line-up is a genuine zero.
    """
    t = live.get(side) if live else None
    if not t or not t.get("Players"):
        return None, None, None, "no FIFA line-up/bookings"
    books = [b for b in (t.get("Bookings") or []) if b.get("IdPlayer")]
    yellows = [b for b in books if b["Card"] == 1]
    reds = [b for b in books if b["Card"] in (2, 3)]
    other = [b["Card"] for b in books if b["Card"] not in (1, 2, 3)]
    for r in (b for b in books if b["Card"] == 3):
        same = [y for y in yellows if y["IdPlayer"] == r["IdPlayer"]]
        if len(same) >= 2:            # drop the second yellow that triggered the dismissal
            exact = [y for y in same if y["Minute"] == r["Minute"]]
            yellows.remove(exact[0] if exact else same[-1])
    status = "ok" if not other else f"unknown card codes {other}"
    return len(yellows), len(reds), len(yellows) + len(reds), status


def fifa_possession(live):
    bp = (live or {}).get("BallPossession") or {}
    h, a = bp.get("OverallHome"), bp.get("OverallAway")
    if h is None or a is None or (h + a) <= 0:
        return None, None
    return 100 * h / (h + a), 100 * a / (h + a)


def fdh_stats(ifes):
    if not ifes:
        return None
    d = fetch_json(FDH_TEAMS.format(ifes=ifes), "fifa_fdh")
    if not d:
        return None
    return {team_id: {k: v for k, v, *_ in vals} for team_id, vals in d.items()}


# ---------------------------------------------------------------- FotMob
def fotmob_find(kickoff, home, away):
    """Find FotMob match id by kickoff (+-36h; FIFA sometimes stores 00:00 for unknown times) and team names
    (either orientation). National teams never meet twice within 36 hours, so names decide."""
    for delta in (0, -1, 1):
        day = (kickoff + dt.timedelta(days=delta)).strftime("%Y%m%d")
        d = fetch_json(FOTMOB_DAY.format(yyyymmdd=day), "fotmob_day")
        for lg in (d or {}).get("leagues", []):
            for m in lg.get("matches", []):
                ut = m.get("status", {}).get("utcTime")
                if not ut or abs((parse_utc(ut) - kickoff).total_seconds()) > 36 * 3600:
                    continue
                h, a = m["home"]["name"], m["away"]["name"]
                if (same_team(home, h) and same_team(away, a)) or (same_team(home, a) and same_team(away, h)):
                    return m["id"], lg.get("name"), lg.get("primaryId"), same_team(home, h)
    return None, None, None, None


_FM_KEYS = {
    "BallPossesion": "possession_pct", "total_shots": "shots_total", "ShotsOnTarget": "shots_on_target",
    "passes": "passes_att", "accurate_passes": "passes_cmp", "Offsides": "offsides",
    "matchstats.headers.tackles": "tackles", "keeper_saves": "saves", "fouls": "fouls_committed",
    "corners": "corners", "yellow_cards": "yellow_cards", "red_cards": "red_cards",
}


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else None


def fotmob_stats(fm_id):
    """Return (home_stats, away_stats, meta) from FotMob match details. Missing keys stay absent."""
    md = fetch_json(FOTMOB_MATCH.format(id=fm_id), "fotmob_match")
    if not md:
        return {}, {}, {}
    status = md.get("header", {}).get("status", {})
    meta = {"reason": (status.get("reason") or {}).get("short"), "awarded": status.get("awarded"),
            "score": status.get("scoreStr"), "league": md.get("general", {}).get("leagueName"),
            "parent_league": md.get("general", {}).get("parentLeagueId"),
            "home": md.get("general", {}).get("homeTeam", {}).get("name"),
            "away": md.get("general", {}).get("awayTeam", {}).get("name"),
            "periods": list(((md.get("content") or {}).get("stats") or {}).get("Periods") or {}) or None}
    home, away = {}, {}
    try:
        groups = md["content"]["stats"]["Periods"]["All"]["stats"]
    except (KeyError, TypeError):
        return home, away, meta
    for g in groups:
        for s in g.get("stats", []):
            field = _FM_KEYS.get(s.get("key"))
            vals = s.get("stats")
            if not field or not vals or len(vals) != 2 or vals[0] is None or vals[1] is None:
                continue
            if field in home:              # keep the first populated occurrence
                continue
            home[field], away[field] = _num(vals[0]), _num(vals[1])
    return home, away, meta


# ---------------------------------------------------------------- ESPN
_ESPN_KEYS = {
    "possessionPct": "possession_pct", "totalShots": "shots_total", "shotsOnTarget": "shots_on_target",
    "totalPasses": "passes_att", "accuratePasses": "passes_cmp", "offsides": "offsides",
    "totalTackles": "tackles", "effectiveTackles": "tackles_won", "saves": "saves",
    "foulsCommitted": "fouls_committed", "wonCorners": "corners", "yellowCards": "yellow_cards",
    "redCards": "red_cards",
}


def espn_find(kickoff, home, away, hint=None):
    order = sorted(ESPN_SLUGS, key=lambda sl: 0 if hint and sl.endswith(hint) else 1)
    for slug in order:
        for delta in (0, -1, 1):
            day = (kickoff + dt.timedelta(days=delta)).strftime("%Y%m%d")
            try:
                d = fetch_json(ESPN_SCOREBOARD.format(league=slug, yyyymmdd=day), "espn_scoreboard")
            except Exception:
                d = None
            for e in (d or {}).get("events", []):
                if abs((parse_utc(e["date"]) - kickoff).total_seconds()) > 6 * 3600:
                    continue
                comps = e["competitions"][0]["competitors"]
                names = {c["homeAway"]: c["team"]["displayName"] for c in comps}
                h, a = names.get("home", ""), names.get("away", "")
                if (same_team(home, h) and same_team(away, a)) or (same_team(home, a) and same_team(away, h)):
                    return e["id"], slug
    return None, None


def espn_stats(ev_id):
    """Return {team_display_name: stats}. All-zero boxes (placeholders) are returned as empty."""
    d = fetch_json(ESPN_SUMMARY.format(id=ev_id), "espn_summary")
    out = {}
    for t in (d or {}).get("boxscore", {}).get("teams", []):
        s = {}
        for x in t.get("statistics", []):
            f = _ESPN_KEYS.get(x["name"])
            if f:
                s[f] = _num(x.get("displayValue"))
        out[t["team"]["displayName"]] = s
    # Treat a box where every value of both teams is zero/None as missing, not as genuine zeros. Cards do not
    # count: some friendly boxes have cards filled from match events while every team statistic, possession
    # included, is still 0 (e.g. Paraguay v Nicaragua and Cabo Verde v Serbia, both June 2026).
    if out and all(not any(v for k, v in s.items() if v is not None and k not in ("yellow_cards", "red_cards"))
                   for s in out.values()):
        return {k: {} for k in out}
    return out


# ---------------------------------------------------------------- assembling records
STAT_FIELDS = ["shots_total", "shots_on_target", "passes_att", "passes_cmp", "possession_pct", "saves",
               "fouls_committed", "tackles", "corners", "offsides", "fk_won_reported", "distance_km",
               "yellow_cards", "red_cards"]


def build_match(kind, id_match, id_stage, kickoff, home_name, away_name, home_id, away_id,
                hs, as_, hp, ap, result_type, competition, stage, wc_number=None):
    """Collect one match from all providers; return two team-perspective records + crosscheck rows."""
    live = fifa_live(id_match)
    ifes = ((live or {}).get("Properties") or {}).get("IdIFES")
    fdh = fdh_stats(ifes) if kind == "WC2026" else None
    fm_id, _, _, fm_same = fotmob_find(kickoff, home_name, away_name)
    fm_h, fm_a, fm_meta = fotmob_stats(fm_id) if fm_id else ({}, {}, {})
    if fm_id and fm_same is False:
        fm_h, fm_a = fm_a, fm_h
    es_id, es_slug = espn_find(kickoff, home_name, away_name, "world" if kind == "WC2026" else "friendly")
    es = espn_stats(es_id) if es_id else {}
    es_h = next((v for k, v in es.items() if same_team(home_name, k)), {})
    es_a = next((v for k, v in es.items() if same_team(away_name, k)), {})
    poss_h, poss_a = fifa_possession(live)
    comp_url = FIFA_MATCH_PAGE.format(comp="17" if kind == "WC2026" else FRIENDLY_COMP,
                                      season=(live or {}).get("IdSeason", ""), stage=id_stage, id=id_match)
    # FIFA ResultType 3 = after extra time; 2 = decided on penalties, which does not say whether extra time came
    # first: friendly tournaments often go straight to penalties after 90 minutes (e.g. Uzbekistan v Iran,
    # 18 Nov 2025: no extra-time periods on FotMob, FIFA numbers the shoot-out kicks from minute 91). FotMob
    # lists extra-time periods whenever they were played (all 9 WC extra-time matches); without FotMob
    # periods, ResultType 2 is taken as extra time (knockout rules).
    periods = fm_meta.get("periods")
    extra_time = result_type == 3 or (result_type == 2 and (periods is None or "FirstExtraHalf" in periods))
    records, cross = [], []
    for side, team, opp, team_id, opp_id, gf, ga, pf, pa, fm_t, fm_o, es_t, es_o, poss in [
        ("HomeTeam", home_name, away_name, home_id, away_id, hs, as_, hp, ap, fm_h, fm_a, es_h, es_a, poss_h),
        ("AwayTeam", away_name, home_name, away_id, home_id, as_, hs, ap, hp, fm_a, fm_h, es_a, es_h, poss_a),
    ]:
        rec = {"team": team, "opponent": opp, "match_type": kind, "competition": competition, "stage": stage,
               "wc_match_number": wc_number, "kickoff_utc": kickoff.strftime("%Y-%m-%dT%H:%MZ"),
               "home_away": "home" if side == "HomeTeam" else "away",
               "goals_for": gf, "goals_against": ga, "pens_for": pf, "pens_against": pa,
               "result_type": result_type, "extra_time": extra_time,
               "nominal_minutes": nominal_minutes(extra_time),
               "awarded": fm_meta.get("awarded"), "fotmob_reason": fm_meta.get("reason"),
               "fifa_match_id": id_match, "fifa_ifes": ifes, "fotmob_id": fm_id, "espn_id": es_id,
               "url_fifa": comp_url,
               "url_fifa_api": FIFA_LIVE.format(id=id_match),
               "url_fdh": FDH_TEAMS.format(ifes=ifes) if fdh else None,
               "url_fotmob": FOTMOB_PAGE.format(id=fm_id) if fm_id else None,
               "url_espn": ESPN_PAGE.format(id=es_id) if es_id else None}
        src = {}

        def put(field, value, source):
            rec[field] = value
            src[field] = source if value is not None else None

        # Passes and possession: FIFA's WC tracking data defines both differently from the providers
        # that cover friendlies (FIFA counts more passes; FIFA possession excludes "in contest" time).
        # One definition is used for every match: FotMob passes (= ESPN) and event-based possession
        # (FIFA live API for friendlies, ESPN/FotMob for WC). FIFA tracking values are kept for reference.
        for f in ["passes_att", "passes_cmp"]:
            if fm_t.get(f) is not None:
                put(f, fm_t[f], "FotMob")
            else:
                put(f, es_t.get(f), "ESPN")
        alt = es_t.get("possession_pct") if es_t.get("possession_pct") is not None else fm_t.get("possession_pct")
        if poss is not None and alt is not None and abs(poss - alt) > 10:
            # FIFA live value contradicts both event providers (seen as 50.0 / 33.3 placeholders): reject it.
            put("possession_pct", alt, f"{'ESPN' if es_t.get('possession_pct') is not None else 'FotMob'} "
                                       f"(FIFA-live {poss:.1f} rejected: >10 pts from ESPN/FotMob)")
        elif poss is not None:
            put("possession_pct", poss, "FIFA-live")
        elif es_t.get("possession_pct") is not None:
            put("possession_pct", es_t["possession_pct"], "ESPN")
        else:
            put("possession_pct", fm_t.get("possession_pct"), "FotMob")
        if fdh and team_id in fdh and opp_id in fdh:
            t, o = fdh[team_id], fdh[opp_id]
            put("shots_total", t.get("AttemptAtGoal"), "FIFA-FDH")
            put("shots_on_target", t.get("AttemptAtGoalOnTarget"), "FIFA-FDH")
            put("fifa_passes_att", t.get("Passes"), "FIFA-FDH")
            put("fifa_passes_cmp", t.get("PassesCompleted"), "FIFA-FDH")
            tp, op = t.get("Possession"), o.get("Possession")
            put("fifa_possession_renorm", 100 * tp / (tp + op) if tp is not None and op else None, "FIFA-FDH")
            put("saves", t.get("GoalkeeperSavesOnTarget"), "FIFA-FDH")
            put("opp_shots_on_target", o.get("AttemptAtGoalOnTarget"), "FIFA-FDH")
            put("fouls_committed", t.get("FoulsAgainst"), "FIFA-FDH")
            put("opp_fouls_committed", o.get("FoulsAgainst"), "FIFA-FDH")
            put("corners", t.get("Corners"), "FIFA-FDH")
            put("offsides", t.get("Offsides"), "FIFA-FDH")
            put("opp_offsides", o.get("Offsides"), "FIFA-FDH")
            put("fk_won_reported", t.get("FreeKicks"), "FIFA-FDH")
            td = t.get("TotalDistance")
            put("distance_km", td / 1000 if td else None, "FIFA-FDH")
            put("time_played_fifa", t.get("TimePlayed"), "FIFA-FDH")
            # Field-level fallback when FDH omits a stat (e.g. match 16 has no GoalkeeperSavesOnTarget).
            for f, src_f, side_t in [("shots_total", "shots_total", "t"), ("shots_on_target", "shots_on_target", "t"),
                                     ("saves", "saves", "t"), ("opp_shots_on_target", "shots_on_target", "o"),
                                     ("fouls_committed", "fouls_committed", "t"),
                                     ("opp_fouls_committed", "fouls_committed", "o"), ("corners", "corners", "t"),
                                     ("offsides", "offsides", "t"), ("opp_offsides", "offsides", "o")]:
                if rec.get(f) is None:
                    fm_x, es_x = (fm_t, es_t) if side_t == "t" else (fm_o, es_o)
                    if fm_x.get(src_f) is not None:
                        put(f, fm_x[src_f], "FotMob (FDH missing)")
                    else:
                        put(f, es_x.get(src_f), "ESPN (FDH missing)")
            tk = fm_t.get("tackles")
            if tk is not None:
                put("tackles", tk, "FotMob")
            else:
                put("tackles", es_t.get("tackles"), "ESPN")
        else:
            # Pre-tournament (or WC without FDH): FotMob primary, ESPN fallback, field by field.
            for f in ["shots_total", "shots_on_target", "saves", "fouls_committed", "tackles", "corners",
                      "offsides"]:
                if fm_t.get(f) is not None:
                    put(f, fm_t[f], "FotMob")
                else:
                    put(f, es_t.get(f), "ESPN")
            for f, of in [("opp_shots_on_target", "shots_on_target"), ("opp_fouls_committed", "fouls_committed"),
                          ("opp_offsides", "offsides")]:
                # Opponent value comes from the same provider as the team's own stat for consistency.
                own = {"opp_shots_on_target": "saves", "opp_fouls_committed": "fouls_committed",
                       "opp_offsides": "offsides"}[f]
                prov = src.get(own) or ("FotMob" if fm_o.get(of) is not None else "ESPN")
                val = fm_o.get(of) if prov == "FotMob" else es_o.get(of)
                put(f, val, prov)
            put("fk_won_reported", None, None)
            put("distance_km", None, None)
        y, r, n, bstat = fifa_bookings(live, side)
        if y is not None:
            put("yellow_cards", y, "FIFA-live")
            put("red_cards", r, "FIFA-live")
        else:
            put("yellow_cards", fm_t.get("yellow_cards"), "FotMob")
            put("red_cards", fm_t.get("red_cards"), "FotMob")
        rec["bookings_status"] = bstat
        for f, s in src.items():
            rec[f"src_{f}"] = s
        records.append(rec)
        cx = {"team": team, "opponent": opp, "kickoff_utc": rec["kickoff_utc"], "match_type": kind,
              "fifa_goals_for": gf, "fotmob_score": fm_meta.get("score"), "fotmob_home": fm_meta.get("home"),
              "fotmob_reason": fm_meta.get("reason"), "fifa_result_type": result_type,
              "fifa_live_possession": poss}
        for f in ["shots_total", "shots_on_target", "passes_att", "passes_cmp", "possession_pct", "saves",
                  "fouls_committed", "tackles", "corners", "offsides", "yellow_cards", "red_cards"]:
            cx[f"primary_{f}"] = rec.get(f)
            cx[f"fotmob_{f}"] = fm_t.get(f)
            cx[f"espn_{f}"] = es_t.get(f)
        cx["espn_tackles_won"] = es_t.get("tackles_won")
        cx["fifa_fk_won"] = rec.get("fk_won_reported")
        cx["derived_fk_won"] = (None if rec.get("opp_fouls_committed") is None or rec.get("opp_offsides") is None
                                else rec["opp_fouls_committed"] + rec["opp_offsides"])
        if fdh and team_id in fdh:
            cx["fifa_fdh_yellow"] = fdh[team_id].get("YellowCards")
            cx["fifa_fdh_red"] = fdh[team_id].get("RedCards")
        cross.append(cx)
    return records, cross


def friendly_list(team_id, opener_kickoff):
    """Completed international friendlies before the opener, newest first."""
    return [m for m in last_matches(team_id, opener_kickoff) if m["IdCompetition"] == FRIENDLY_COMP]


def last_matches(team_id, opener_kickoff):
    """Completed matches of any competition before the opener, newest first (FIFA team form)."""
    to = (opener_kickoff - dt.timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tf = fetch_json(FIFA_TEAMFORM.format(team=team_id, to=to, n=60), "fifa_teamform")
    out = [m for m in tf["MatchesList"]
           if parse_utc(m["Date"]) < opener_kickoff
           and m.get("HomeTeamScore") is not None and m.get("AwayTeamScore") is not None]
    out.sort(key=lambda m: m["Date"], reverse=True)
    return out


def friendly_job(fname, m, rank):
    """Job for one friendly; only the perspective of fname is kept."""
    h, a = tname(m["Home"]), tname(m["Away"])
    return (f"FRIENDLY {fname} #{rank}: {m['Date'][:10]} {h} v {a}", fname,
            ("FRIENDLY2026", m["IdMatch"], m["IdStage"], parse_utc(m["Date"]), h, a,
             m["Home"]["IdTeam"], m["Away"]["IdTeam"], m["HomeTeamScore"],
             m["AwayTeamScore"], m.get("HomeTeamPenaltyScore"), m.get("AwayTeamPenaltyScore"),
             m.get("ResultType"), "International friendly", "Friendly", None))


def fotmob_team_id(records, fname):
    """FotMob team id of fname, read from the FotMob details of one of its WC matches."""
    for r in records:
        if r["team"] == fname and r["match_type"] == "WC2026" and r["fotmob_id"]:
            g = fetch_json(FOTMOB_MATCH.format(id=r["fotmob_id"]), "fotmob_match")["general"]
            for side in ("homeTeam", "awayTeam"):
                if same_team(fname, g[side]["name"]):
                    return g[side]["id"]
    return None


def fotmob_recent(fm_team, before, n=3):
    """The n newest finished matches before `before` in FotMob's team fixtures, newest first."""
    d = fetch_json(FOTMOB_TEAM.format(id=fm_team), "fotmob_team")
    fx = (((d or {}).get("fixtures") or {}).get("allFixtures") or {}).get("fixtures") or []
    done = [x for x in fx if (x.get("status") or {}).get("finished") and (x.get("status") or {}).get("utcTime")
            and parse_utc(x["status"]["utcTime"]) < before]
    done.sort(key=lambda x: x["status"]["utcTime"], reverse=True)
    return done[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default=None, help="comma-separated workbook team names (pilot)")
    args = ap.parse_args()

    rows = load_workbook_rows()
    cal = load_calendar()
    wb_teams = sorted({r["team"] for r in rows})
    want = set(t.strip() for t in args.teams.split(",")) if args.teams else set(wb_teams)

    # FIFA team id and opener kickoff per workbook team.
    team_id, opener = {}, {}
    for num in sorted(cal, key=lambda n: cal[n]["Date"]):
        m = cal[num]
        for s in ("Home", "Away"):
            fname = tname(m[s])
            team_id.setdefault(fname, m[s]["IdTeam"])
            opener.setdefault(fname, parse_utc(m["Date"]))

    from concurrent.futures import ThreadPoolExecutor
    records, cross, list_rows = [], [], []
    jobs = []                                  # (label, keep_team or None, kwargs)
    # World Cup matches involving wanted teams.
    for num in sorted(cal):
        m = cal[num]
        h, a = tname(m["Home"]), tname(m["Away"])
        wb_names = {WB_TO_FIFA.get(t, t) for t in want}
        if h not in wb_names and a not in wb_names:
            continue
        jobs.append((f"WC {num:>3} {h} {m['HomeTeamScore']}-{m['AwayTeamScore']} {a}", None,
                     ("WC2026", m["IdMatch"], m["IdStage"], parse_utc(m["Date"]), h, a,
                      m["Home"]["IdTeam"], m["Away"]["IdTeam"], m["HomeTeamScore"], m["AwayTeamScore"],
                      m.get("HomeTeamPenaltyScore"), m.get("AwayTeamPenaltyScore"), m.get("ResultType"),
                      "FIFA World Cup 2026", m["StageName"][0]["Description"], num)))

    def run(job):
        label, keep, a = job
        return label, keep, build_match(*a[:15], wc_number=a[15])

    def run_all(batch):
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, (label, keep, (rec, cx)) in enumerate(pool.map(run, batch), 1):
                for r_, c_ in zip(rec, cx):
                    # Friendlies: keep only the wanted team's perspective (the opponent row feeds no row).
                    if keep is None or r_["team"] == keep:
                        records.append(r_)
                        cross.append(c_)
                print(f"[{i}/{len(batch)}] {label}", flush=True)

    run_all(jobs)

    # Openers (user decision 2026-10-16): each team's last international friendly before its opener. If that
    # friendly has no published team statistics on any source, the next older friendly is collected, until one
    # has statistics. No age limit (user decision 2026-10-16: Czechia goes back to Sep 2025 and Iran to Nov
    # 2025). prepare.py uses the newest collected friendly with statistics.
    fr_lists, last_any = {}, {}
    for wb in sorted(want):
        fname = WB_TO_FIFA.get(wb, wb)
        fr_lists[fname] = friendly_list(team_id[fname], opener[fname])
        last_any[fname] = (last_matches(team_id[fname], opener[fname]) or [None])[0]
        print(f"FRIENDLY {fname}: {len(fr_lists[fname])} friendlies before the opener, last "
              f"{fr_lists[fname][0]['Date'][:10] if fr_lists[fname] else 'none found'}", flush=True)
    collected = {f: 0 for f in fr_lists}        # friendlies collected per team, newest first
    pending = [f for f in fr_lists if fr_lists[f]]
    while pending:
        run_all([friendly_job(f, fr_lists[f][collected[f]], collected[f] + 1) for f in pending])
        nxt = []
        for f in pending:
            m = fr_lists[f][collected[f]]
            collected[f] += 1
            rec = next(r for r in records if r["team"] == f and r["fifa_match_id"] == m["IdMatch"])
            if all(rec.get(x) is None for x in STATS_PRESENCE):
                if collected[f] < len(fr_lists[f]):
                    print(f"FRIENDLY {f}: {m['Date'][:10]} v {rec['opponent']} has no team statistics; "
                          "collecting the next older friendly", flush=True)
                    nxt.append(f)
                else:
                    print(f"FRIENDLY {f}: no friendly with team statistics found", flush=True)
        pending = nxt

    for f in sorted(fr_lists):
        x = last_any[f]
        if x is not None:
            list_rows.append({"team": f, "source": "FIFA teamform (any competition)", "rank": 1,
                              "collected": None, "utc": x["Date"][:16], "home": tname(x["Home"]),
                              "away": tname(x["Away"]), "score": f'{x["HomeTeamScore"]}-{x["AwayTeamScore"]}',
                              "competition": "International friendly" if x["IdCompetition"] == FRIENDLY_COMP
                              else f'FIFA competition {x["IdCompetition"]}', "result_type": x.get("ResultType"),
                              "fifa_match_id": x["IdMatch"], "fotmob_id": None})
        for rank, m in enumerate(fr_lists[f][:max(N_FR_LIST, collected[f])], 1):
            list_rows.append({"team": f, "source": "FIFA teamform (friendlies)", "rank": rank,
                              "collected": rank <= collected[f], "utc": m["Date"][:16], "home": tname(m["Home"]),
                              "away": tname(m["Away"]), "score": f'{m["HomeTeamScore"]}-{m["AwayTeamScore"]}',
                              "competition": "International friendly", "result_type": m.get("ResultType"),
                              "fifa_match_id": m["IdMatch"], "fotmob_id": None})

    # FotMob team fixtures as an independent check on FIFA's friendly lists: a friendly that FIFA omits would
    # show up here as a newer match before the opener.
    for f in sorted(fr_lists):
        fm_team = fotmob_team_id(records, f)
        if fm_team is None:
            print(f"FOTMOB {f}: team id not found", flush=True)
            continue
        for rank, x in enumerate(fotmob_recent(fm_team, opener[f]), 1):
            list_rows.append({"team": f, "source": "FotMob team fixtures", "rank": rank, "collected": None,
                              "utc": x["status"]["utcTime"][:16], "home": x["home"]["name"],
                              "away": x["away"]["name"], "score": x["status"].get("scoreStr"),
                              "competition": (x.get("tournament") or {}).get("name"), "result_type": None,
                              "fifa_match_id": None, "fotmob_id": x.get("id")})

    write_csv(RAW_CSV, records)
    write_csv(INTERMEDIATE / "crosscheck.csv", cross)
    write_csv(INTERMEDIATE / "friendly_lists.csv", list_rows)
    print(f"wrote {len(records)} raw records")


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
