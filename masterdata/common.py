"""Shared helpers: cached HTTP fetch, paths, team-name aliases, minutes."""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # data_wrangling/
CACHE = ROOT / "raw_cache"                            # downloaded JSON (re-fetchable)
INPUT = ROOT / "input"                                # original workbook, never written
OUTPUT = ROOT / "output"                              # final deliverables
INTERMEDIATE = ROOT / "intermediate"                  # CSVs from collect.py, read by prepare.py
WORKBOOK_IN = INPUT / "data_wrangling.xlsx"
WORKBOOK_OUT = OUTPUT / "data_wrangling_filled.xlsx"
RAW_CSV = INTERMEDIATE / "raw_history.csv"

UA = {"User-Agent": "Mozilla/5.0 (CDU assignment data collection)"}

WC_COMP, WC_SEASON = "17", "285023"
FIFA_CALENDAR = (f"https://api.fifa.com/api/v3/calendar/matches?idCompetition={WC_COMP}"
                 f"&idSeason={WC_SEASON}&count=200&language=en")
FIFA_LIVE = "https://api.fifa.com/api/v3/live/football/{id}?language=en"
FIFA_TEAMFORM = "https://api.fifa.com/api/v3/teamform/{team}?to={to}&count={n}&language=en"
FDH_TEAMS = "https://fdh-api.fifa.com/v1/stats/match/{ifes}/teams.json"
FIFA_MATCH_PAGE = "https://www.fifa.com/en/match-centre/match/{comp}/{season}/{stage}/{id}"
FOTMOB_DAY = "https://www.fotmob.com/api/data/matches?date={yyyymmdd}"
FOTMOB_MATCH = "https://www.fotmob.com/api/data/matchDetails?matchId={id}"
FOTMOB_PAGE = "https://www.fotmob.com/match/{id}"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={yyyymmdd}"
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/all/summary?event={id}"
ESPN_PAGE = "https://www.espn.com/soccer/match/_/gameId/{id}"
FIFA_RANKING = "https://inside.fifa.com/api/ranking-overview?locale=en&dateId={date_id}"
FIFA_RANKING_PAGE = "https://inside.fifa.com/fifa-rankings/world-ranking/men"
RANKING_CSV = INTERMEDIATE / "fifa_ranking.csv"

# Workbook spelling -> FIFA spelling (only names that differ).
WB_TO_FIFA = {"Turkey": "Türkiye", "Curacao": "Curaçao", "Ivory Coast": "Côte d'Ivoire"}


def fetch_json(url, source, ttl_days=None, sleep=0.4):
    """GET url as JSON, caching the raw body under raw_cache/<source>/.

    Returns None on HTTP 404 (cached as a 404 marker so it is not re-requested).
    """
    folder = CACHE / source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")
    if path.exists() and (ttl_days is None or time.time() - path.stat().st_mtime < ttl_days * 86400):
        rec = json.loads(path.read_text())
        return rec["body"]
    req = urllib.request.Request(url, headers=UA)
    body = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                body = None
                break
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    path.write_text(json.dumps({"url": url, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "body": body}))
    time.sleep(sleep)
    return body


def norm_name(s):
    """Loose team-name key for matching across providers."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    for a, b in [("&", "and"), ("-", " "), (".", ""), ("'", "")]:
        s = s.replace(a, b)
    return " ".join(s.split())


# Provider spellings that differ from FIFA's, keyed by normalised FIFA name.
ALIASES = {
    "korea republic": {"south korea", "korea republic", "republic of korea"},
    "ir iran": {"iran", "ir iran"},
    "congo dr": {"dr congo", "congo dr", "democratic republic of the congo", "congo democratic republic"},
    "cabo verde": {"cape verde", "cabo verde", "cape verde islands"},
    "turkiye": {"turkiye", "turkey"},
    "cote divoire": {"ivory coast", "cote divoire"},
    "curacao": {"curacao"},
    "usa": {"usa", "united states", "united states of america"},
    "czechia": {"czechia", "czech republic"},
    "bosnia and herzegovina": {"bosnia and herzegovina", "bosnia herzegovina", "bosnia"},
    "korea dpr": {"north korea", "korea dpr"},
    "china pr": {"china", "china pr"},
    "kyrgyz republic": {"kyrgyzstan", "kyrgyz republic"},
    "chinese taipei": {"chinese taipei", "taiwan"},
    "hong kong, china": {"hong kong", "hong kong china"},
    "st kitts and nevis": {"st kitts and nevis", "saint kitts and nevis"},
    "st lucia": {"st lucia", "saint lucia"},
    "st vincent and the grenadines": {"st vincent and the grenadines", "saint vincent and the grenadines",
                                      "st vincent grenadines"},
    "eswatini": {"eswatini", "swaziland"},
    "the gambia": {"gambia", "the gambia"},
    "guinea bissau": {"guinea bissau"},
    "sao tome and principe": {"sao tome and principe", "sao tome e principe"},
    "central african republic": {"central african republic", "car"},
    "republic of ireland": {"ireland", "republic of ireland"},
    "northern ireland": {"northern ireland"},
    "north macedonia": {"north macedonia", "macedonia"},
    "us virgin islands": {"us virgin islands", "united states virgin islands"},
    "british virgin islands": {"british virgin islands"},
    "turks and caicos islands": {"turks and caicos", "turks and caicos islands"},
    "timor leste": {"timor leste", "east timor"},
    "brunei darussalam": {"brunei", "brunei darussalam"},
    "united arab emirates": {"uae", "united arab emirates"},
    "syria": {"syria", "syrian arab republic"},
    "palestine": {"palestine"},
    "trinidad and tobago": {"trinidad and tobago", "trinidad & tobago"},
    "antigua and barbuda": {"antigua and barbuda"},
    "dominican republic": {"dominican republic"},
    "congo": {"congo", "republic of congo", "congo republic"},
}


def same_team(fifa_name, other_name):
    a, b = norm_name(fifa_name), norm_name(other_name)
    if a == b:
        return True
    return b in ALIASES.get(a, set())


def nominal_minutes(extra_time):
    return 120 if extra_time else 90
