# Data wrangling code

Scripts that fill the World Cup 2026 workbook (`input/data_wrangling.xlsx`) with pre-match predictors for every team in every match, and write the result to `output/data_wrangling_filled.xlsx`.

Each row's predictors come from exactly one previous match of that team:

- Opener row: the team's last international friendly before the tournament. If that friendly has no published team statistics, the most recent friendly that has them.
- Later rows: the team's previous World Cup 2026 match.
- `FIFA ranking points` is the exception. It is the team's total points in the last official FIFA ranking published before kickoff.

## Requirements

- Python 3.9 or newer
- Packages: `pandas`, `numpy`, `openpyxl`
- Internet access on the first run (data is downloaded from public web APIs)

```bash
pip install pandas numpy openpyxl
```

## Folder layout

```
data_wrangling/
├── code/            scripts (this folder)
├── input/           original workbook, read only: data_wrangling.xlsx
├── raw_cache/       downloaded JSON responses, created automatically
├── intermediate/    CSV files written by collect.py and collect_ranking.py
└── output/          final workbook and summary files
```

All paths are resolved from the location of the scripts, so the scripts can be started from any directory. The examples below assume you are inside `code/`.

## How to run

Run the scripts in this order.

### 1. Collect match statistics

```bash
python3 collect.py
```

Downloads the World Cup 2026 fixtures and results, then per-match team statistics for every World Cup match and for each team's friendlies before its opener. Statistics come from FIFA, FotMob and ESPN; values from each provider are kept side by side.

Writes to `intermediate/`:

| file | content |
|---|---|
| `raw_history.csv` | one row per team per match, with the chosen value, its source and URL for every field |
| `crosscheck.csv` | the same matches with every provider's value side by side |
| `friendly_lists.csv` | each team's newest friendlies (FIFA) and newest matches before the opener (FotMob) |

To test on a few teams first:

```bash
python3 collect.py --teams "Mexico,South Africa"
```

Team names must be spelled as in the workbook.

### 2. Collect FIFA ranking points

```bash
python3 collect_ranking.py
```

Downloads three official FIFA men's ranking releases for the 48 World Cup teams and checks that each release's previous points equal the preceding release's total points.

Writes `intermediate/fifa_ranking.csv`.

### 3. Check the collected data (optional, recommended)

```bash
python3 checks.py
```

Prints consistency checks on the collected data. Read the output before continuing:

- number of records and link coverage per provider
- field coverage by match type
- agreement between providers (exact-match rate and largest difference)
- FIFA scores compared with FotMob scores
- extra-time and awarded matches
- FIFA friendly list compared with FotMob team fixtures

### 4. Build the workbook

```bash
python3 prepare.py
```

Reads the CSV files from `intermediate/` and the original workbook, calculates the predictors, and writes the filled workbook. The original workbook is never modified.

Writes to `output/`:

| file | content |
|---|---|
| `data_wrangling_filled.xlsx` | the filled workbook (sheet `in`) plus supporting sheets |
| `COMPLETENESS.md` | how many cells are filled per column, which source supplied each value, decisions made, reasons for blank cells |

`prepare.py` finishes with a verification step. It prints `verification passed` when all checks succeed (row count, match numbers, goals against FIFA, every source match before kickoff, one source match per row, saving between 0 and 100, ranking points).

`prepare.py --teams "Mexico,South Africa"` fills only the listed teams.

## Output workbook

Sheet `in` keeps the layout of the original workbook. Predictor columns are filled, `act_goals` holds the team's goals in the match, and `pred_goals` is left blank. The last column, `previous_match_duration`, is the length of the source match (90, or 120 after extra time).

| column | meaning |
|---|---|
| `free_kicks_won_per90` | opponent fouls + opponent offsides, per 90 minutes |
| `Shot accurrancy` | % of shots on target |
| `Pass acurracy` | % of passes completed |
| `possession` | % ball possession |
| `saving` | % of opponent shots on target saved (0 when the opponent had none; capped at 100) |
| `fouls_committed_per90` | fouls committed, per 90 minutes |
| `tackles_attempted_per90` | tackles attempted, per 90 minutes |
| `corners_per90` | corners won, per 90 minutes |
| `FIFA ranking points` | total points in the last FIFA ranking before kickoff |
| `goals_conceded_per90` | goals conceded in the source match, per 90 minutes |
| `cards_per90` | yellow + red cards, per 90 minutes |
| `shots_per90` | total shots, per 90 minutes |

Supporting sheets:

| sheet | content |
|---|---|
| `raw_history` | all collected matches; `used_in_rows` shows which workbook rows use each match |
| `row_windows` | the source match chosen for each row |
| `definitions` | formula, unit and source of every column |
| `sources` | every data source, endpoint and what it was used for |
| `missing_notes` | each blank cell and the reason |
| `match_check` | workbook teams, dates and scores compared with the FIFA calendar |
| `freekick_validation` | check of the derived free-kick measure against FIFA's reported value |
| `crosscheck` | provider values side by side |
| `friendly_lists` | friendly candidates per team |
| `fifa_ranking` | ranking releases used |

## Script overview

| script | role |
|---|---|
| `common.py` | shared settings: paths, API URLs, team-name aliases, cached download function `fetch_json` |
| `collect.py` | downloads match data and writes the raw CSV files |
| `collect_ranking.py` | downloads FIFA ranking points |
| `checks.py` | prints consistency checks on the collected data |
| `prepare.py` | calculates the predictors and writes the filled workbook |

## Notes

- **Caching.** Every download is saved under `raw_cache/<source>/`. Later runs reuse the saved responses, so they are fast and work offline. To download a source again, delete its folder inside `raw_cache/`.
- **Blank cells.** If the chosen source match lacks a statistic on every provider, the cell stays blank. An older match is never used as a substitute. The reason for each blank cell is in `missing_notes` and `COMPLETENESS.md`.
- **Per 90 minutes.** Counts are scaled to 90 minutes. Counts from a match that went to extra time are multiplied by 90/120. Percentages are the match's own ratio.
- **Team names.** Workbook spellings that differ from FIFA's (`Turkey`, `Curacao`, `Ivory Coast`) are mapped in `common.py` (`WB_TO_FIFA`). Spellings used by other providers are in `ALIASES`.
- **Ranking releases.** The three release ids are listed at the top of `collect_ranking.py` (`RELEASES`).
- **Dates.** Dates in the original workbook are read as day/month/year. Dates that Excel stored with day and month swapped are corrected, and every correction is logged in `match_check`.
