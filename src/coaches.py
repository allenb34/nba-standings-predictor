"""Head coach per team-season from Basketball-Reference, cached -> data/raw/coaches_{end_year}.csv,
plus data/processed/coaches.csv with one row per (season_end, team): the coach who STARTED the season.

v2 exploratory (branch v2-coach-effects). Does not touch the locked v1 pipeline.
"""
import time

import pandas as pd
import requests
from lxml import html as lxml_html

from fetch import HTML_CACHE, RAW, _retry
from features import BBREF_TO_NBA, PROCESSED

YEARS = range(2012, 2028)  # 2011-12 .. 2026-27


def _html(end_year: int) -> str:
    path = HTML_CACHE / f"NBA_{end_year}_coaches.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_coaches.html"

    def get():
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research; nba-standings-predictor)"})
        if r.status_code == 429:
            raise RuntimeError("429 rate limited")
        r.raise_for_status()
        return r.content.decode("utf-8")

    text = _retry(get, f"bbref coaches {end_year}")
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(4.0)  # BBRef allows ~20 req/min
    return text


def coaches_season(end_year: int) -> pd.DataFrame:
    out = RAW / f"coaches_{end_year}.csv"
    if out.exists():
        return pd.read_csv(out)
    table = lxml_html.fromstring(_html(end_year)).get_element_by_id("NBA_coaches")
    rows = []
    for order, tr in enumerate(table.xpath("./tbody/tr[not(contains(@class,'thead'))]")):
        cells = {c.get("data-stat"): c for c in tr.xpath("./th|./td")}
        link = cells["coach"].xpath(".//a/@href")
        rows.append({
            "season_end": end_year,
            "row_order": order,
            "coach_id": link[0].rsplit("/", 1)[-1].removesuffix(".html"),
            "coach": cells["coach"].text_content().strip(),
            "team_bbref": cells["team"].text_content().strip(),
            "seasons_overall": int(cells["seas_num_overall"].text_content()),
            "games": int(cells["cur_g"].text_content() or 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"  coaches {end_year}: {len(df)} rows")
    return df


def main():
    raw = pd.concat([coaches_season(y) for y in YEARS], ignore_index=True)
    raw["team"] = raw["team_bbref"].map(lambda t: BBREF_TO_NBA.get(t, t))
    raw = raw.sort_values(["season_end", "row_order"])

    # Rows within a team must be chronological, not alphabetical: Fizdale (19 G) was fired and replaced by
    # Bickerstaff in 2017-18 MEM; Green (12 G) was replaced by Borrego in 2025-26 NOP.
    first = raw.groupby(["season_end", "team"])["coach"].first()
    assert first[(2018, "MEM")] == "David Fizdale" and first[(2026, "NOP")] == "Willie Green", "coach rows not chronological"

    g = raw.groupby(["season_end", "team"])
    starters = g.first()[["coach_id", "coach", "seasons_overall", "games"]].rename(columns={"games": "starter_games"})
    starters["n_coaches"] = g.size()
    starters["midseason_change"] = starters["n_coaches"] > 1
    starters = starters.reset_index()
    assert (starters.groupby("season_end")["team"].nunique() == 30).all()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    starters.to_csv(PROCESSED / "coaches.csv", index=False)
    hist = starters[starters["season_end"] < 2027]
    print(f"wrote coaches.csv: {len(starters)} team-seasons, {hist['coach_id'].nunique()} distinct coaches 2011-12..2025-26, "
          f"{hist['midseason_change'].sum()} team-seasons with a midseason change")


if __name__ == "__main__":
    main()
