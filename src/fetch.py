"""All network pulls, cache-first. A second run makes zero network calls.

- Basketball-Reference advanced tables, 2009-10 .. 2025-26 -> data/raw/bbref_advanced_{end_year}.csv
- nba_api team advanced stats, 2010-11 .. 2025-26         -> data/raw/team_advanced_{season}.csv
- nba_api CommonTeamRoster, 2026-27                         -> data/raw/rosters_2026-27.csv
"""
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # this machine's TLS-inspecting proxy breaks certifi

import pandas as pd
import requests
from lxml import html as lxml_html
from nba_api.stats.endpoints import commonteamroster, leaguedashteamstats
from nba_api.stats.static import teams as static_teams

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HTML_CACHE = RAW / "html"  # gitignored; parsed CSVs are the committed cache

BBREF_YEARS = range(2010, 2027)  # end years: 2009-10 .. 2025-26
TEAM_SEASONS = [f"{y - 1}-{str(y)[2:]}" for y in range(2011, 2027)]  # 2010-11 .. 2025-26
ROSTER_SEASON = "2026-27"

network_calls = 0


def _retry(fn, context, tries=5):
    global network_calls
    for attempt in range(1, tries + 1):
        network_calls += 1
        try:
            return fn()
        except Exception as e:  # network errors, timeouts, nba_api JSON decode errors
            print(f"  {context}: attempt {attempt}/{tries} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(min(5 * 2 ** attempt, 60))


# ---------- Basketball-Reference ----------

def _bbref_html(end_year: int) -> str:
    path = HTML_CACHE / f"NBA_{end_year}_advanced.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_advanced.html"

    def get():
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research; nba-standings-predictor)"})
        if r.status_code == 429:
            raise RuntimeError("429 rate limited")
        r.raise_for_status()
        return r.content.decode("utf-8")  # requests guesses latin-1 and mangles accents

    text = _retry(get, f"bbref {end_year}")
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(4.0)  # BBRef allows ~20 req/min
    return text


def parse_bbref(text: str, end_year: int) -> pd.DataFrame:
    """One row per (player, team row) in page order; TOT/2TM rows kept with team='TOT'."""
    doc = lxml_html.fromstring(text)
    table = doc.get_element_by_id("advanced")
    rows = []
    for order, tr in enumerate(table.xpath("./tbody/tr[not(contains(@class,'thead'))]")):
        cells = {td.get("data-stat"): td for td in tr.xpath("./th|./td")}
        name_td = cells.get("name_display") if "name_display" in cells else cells.get("player")
        team_td = cells.get("team_name_abbr") if "team_name_abbr" in cells else cells.get("team_id")
        if name_td is None or team_td is None or not name_td.get("data-append-csv"):  # skips League Average
            continue
        team = team_td.text_content().strip()
        if team.endswith("TM"):  # newer pages use 2TM/3TM for the season total
            team = "TOT"

        def num(stat):
            td = cells.get(stat)
            s = td.text_content().strip() if td is not None else ""
            return float(s) if s else float("nan")

        rows.append({
            "season_end": end_year,
            "row_order": order,
            "player_id": name_td.get("data-append-csv"),
            "player": name_td.text_content().strip().rstrip("*"),
            "team_bbref": team,
            "age": num("age"),
            "mp": num("mp"),
            "bpm": num("bpm"),
        })
    df = pd.DataFrame(rows)
    assert df["player_id"].notna().all(), f"{end_year}: missing player slugs"
    return df


def bbref_advanced(end_year: int) -> pd.DataFrame:
    out = RAW / f"bbref_advanced_{end_year}.csv"
    if out.exists():
        return pd.read_csv(out)
    df = parse_bbref(_bbref_html(end_year), end_year)
    df.to_csv(out, index=False)
    print(f"  bbref {end_year}: {len(df)} rows")
    return df


# ---------- nba_api ----------

def team_advanced(season: str) -> pd.DataFrame:
    out = RAW / f"team_advanced_{season}.csv"
    if out.exists():
        return pd.read_csv(out)
    df = _retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced", timeout=60).get_data_frames()[0], f"team stats {season}")
    df = df[["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "NET_RATING"]].assign(season=season)
    assert len(df) == 30, f"{season}: {len(df)} teams"
    df.to_csv(out, index=False)
    print(f"  team stats {season}: ok")
    time.sleep(1.0)
    return df


def rosters_2026_27() -> pd.DataFrame:
    out = RAW / f"rosters_{ROSTER_SEASON}.csv"
    if out.exists():
        return pd.read_csv(out)
    frames = []
    for t in static_teams.get_teams():
        df = _retry(lambda: commonteamroster.CommonTeamRoster(
            team_id=t["id"], season=ROSTER_SEASON, timeout=60).get_data_frames()[0], f"roster {t['abbreviation']}")
        frames.append(df[["TeamID", "PLAYER", "PLAYER_ID", "AGE", "BIRTH_DATE", "EXP"]]
                      .assign(team=t["abbreviation"]))
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"  rosters {ROSTER_SEASON}: {len(df)} players")
    return df


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for y in BBREF_YEARS:
        bbref_advanced(y)
    for s in TEAM_SEASONS:
        team_advanced(s)
    rosters_2026_27()
    print(f"network calls this run: {network_calls}")


if __name__ == "__main__":
    main()
