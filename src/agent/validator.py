"""
Fetch real FIFA match results via OpenAI web search.
Saves output to data/actuals/YYYY-MM-DD_actuals.json.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parents[2] / ".env")

logger = logging.getLogger(__name__)

ACTUALS_DIR = Path(__file__).parents[2] / "data" / "actuals"
LOGS_DIR = Path(__file__).parents[2] / "logs"
ACTUALS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class MatchQuery(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    date: str  # YYYY-MM-DD


class ActualResult(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result: Literal["H", "D", "A"]
    source_url: str
    confidence: Literal["high", "medium", "low"]


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def _parse_scores_from_text(text: str, home: str, away: str) -> tuple[int, int] | None:
    """Extract score from web search response text. Returns (home_goals, away_goals) or None."""
    import re

    # Look for patterns like "2-1", "Brazil 2-1 Argentina", "2 : 1"
    patterns = [
        rf"{re.escape(home)}\s+(\d+)\s*[-–:]\s*(\d+)\s+{re.escape(away)}",
        rf"(\d+)\s*[-–:]\s*(\d+)",
        rf"(\d+)\s*-\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                g1, g2 = int(match.group(1)), int(match.group(2))
                if g1 <= 20 and g2 <= 20:  # sanity check
                    return g1, g2
            except (ValueError, IndexError):
                continue
    return None


def _fetch_single_result(client: OpenAI, query: MatchQuery, log_fh) -> ActualResult | None:
    """Fetch result for one match via OpenAI web search. Returns None if not found."""
    import re

    date_parts = query.date.split("-")
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month_str = month_names[int(date_parts[1])] if len(date_parts) >= 2 else date_parts[1]
    day_str = date_parts[2].lstrip("0") if len(date_parts) >= 3 else date_parts[2]
    year_str = date_parts[0]

    prompt = (
        f"Search for the FINAL SCORE of this football match that was played on "
        f"{month_str} {day_str}, {year_str}:\n"
        f"{query.home_team} vs {query.away_team} ({query.tournament})\n\n"
        f"RULES:\n"
        f"- ONLY report the score if you find it confirmed on a real sports website "
        f"(FIFA.com, BBC Sport, ESPN, Sky Sports, etc.).\n"
        f"- If the match has NOT been played yet, or you cannot find a verified result, "
        f"respond with exactly: RESULT_NOT_FOUND\n"
        f"- Do NOT guess, estimate, or make up any score.\n"
        f"- Do NOT report a score if the match is still in progress.\n"
        f"- If found, state the final score clearly, e.g. '{query.home_team} 2-1 {query.away_team}' "
        f"and include the source URL."
    )

    last_exc = None
    for attempt in range(2):
        try:
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            output_text = response.output_text or ""

            log_fh.write(f"\n=== QUERY: {query.home_team} vs {query.away_team} {query.date} ===\n")
            log_fh.write(f"RESPONSE:\n{output_text}\n")

            # Explicit not-found signal
            if "RESULT_NOT_FOUND" in output_text:
                logger.info(
                    "No result yet for %s vs %s on %s (agent confirmed not found).",
                    query.home_team, query.away_team, query.date,
                )
                return None

            # Also treat "not been played", "yet to be played", "upcoming" as not found
            lower = output_text.lower()
            not_played_phrases = [
                "not been played", "yet to be played", "upcoming", "has not taken place",
                "no result", "not yet played", "scheduled", "not available",
            ]
            if any(p in lower for p in not_played_phrases):
                logger.info(
                    "Match %s vs %s on %s not yet played (detected in response).",
                    query.home_team, query.away_team, query.date,
                )
                return None

            # Try to extract score
            scores = _parse_scores_from_text(output_text, query.home_team, query.away_team)

            if scores is None:
                logger.warning(
                    "Could not parse score for %s vs %s on %s — skipping (not storing guesses).",
                    query.home_team, query.away_team, query.date,
                )
                return None  # Return None, not a fake 0-0

            h_goals, a_goals = scores

            # Sanity check: score > 10 is almost certainly hallucinated for a WC match
            if h_goals > 10 or a_goals > 10:
                logger.warning(
                    "Suspicious score %d-%d for %s vs %s — likely hallucinated, skipping.",
                    h_goals, a_goals, query.home_team, query.away_team,
                )
                return None

            if h_goals > a_goals:
                result_code: Literal["H", "D", "A"] = "H"
            elif a_goals > h_goals:
                result_code = "A"
            else:
                result_code = "D"

            # Extract source URL
            source_url = ""
            url_match = re.search(r"https?://\S+", output_text)
            if url_match:
                source_url = url_match.group(0).rstrip(")")

            # Confidence based on explicit confirmation keywords
            if any(kw in lower for kw in ["final score", "full time", "ft:", "full-time", "result:"]):
                confidence: Literal["high", "medium", "low"] = "high"
            elif source_url:
                confidence = "medium"
            else:
                confidence = "low"

            return ActualResult(
                date=query.date,
                home_team=query.home_team,
                away_team=query.away_team,
                home_score=h_goals,
                away_score=a_goals,
                result=result_code,
                source_url=source_url,
                confidence=confidence,
            )

        except Exception as exc:
            last_exc = exc
            logger.warning("OpenAI call failed (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(10)

    logger.error("Skipping %s vs %s after 2 failed attempts: %s", query.home_team, query.away_team, last_exc)
    return None


def update_completed_results(schedule: dict, today: str) -> int:
    """
    Fetch results for all matches where date <= today and not yet in completed_matches_results.json.
    Appends new confirmed results to the file. Returns count of new results added.
    schedule: {date: [(home, away, tournament), ...]}
    today: YYYY-MM-DD string
    """
    completed_path = ACTUALS_DIR / "completed_matches_results.json"

    existing: list[dict] = []
    if completed_path.exists():
        with open(completed_path) as f:
            existing = json.load(f)

    # Track already-fetched matches (skip low-confidence re-fetches too)
    fetched: set[tuple[str, str, str]] = {
        (r["home_team"], r["away_team"], r["date"]) for r in existing
    }

    to_fetch: list[MatchQuery] = []
    for date_str, matches in sorted(schedule.items()):
        if date_str >= today:
            break  # only fetch strictly past dates — today's matches may not have finished
        for home, away, tournament in matches:
            if (home, away, date_str) not in fetched:
                to_fetch.append(MatchQuery(
                    home_team=home, away_team=away,
                    tournament=tournament, date=date_str,
                ))

    if not to_fetch:
        logger.info("completed_matches_results.json is up to date — no new matches to fetch.")
        return 0

    logger.info("Fetching %d new match results (dates up to %s)...", len(to_fetch), today)
    client = _get_client()
    log_path = LOGS_DIR / f"validator_live_{today}.log"
    new_results: list[ActualResult] = []

    with open(log_path, "a") as log_fh:
        for query in to_fetch:
            result = _fetch_single_result(client, query, log_fh)
            if result and result.confidence != "low":
                new_results.append(result)
                logger.info(
                    "+ %s vs %s → %d-%d (%s) [%s]",
                    result.home_team, result.away_team,
                    result.home_score, result.away_score,
                    result.result, result.confidence,
                )
            time.sleep(1)  # rate-limit between searches

    if new_results:
        updated = existing + [r.model_dump() for r in new_results]
        with open(completed_path, "w") as f:
            json.dump(updated, f, indent=2)
        logger.info(
            "Saved completed_matches_results.json — +%d new results (total=%d).",
            len(new_results), len(updated),
        )

    return len(new_results)


def run(matches: list[MatchQuery], date: str) -> list[ActualResult]:
    """
    Fetch real results for all matches on a given date.
    Saves to data/actuals/{date}_actuals.json.
    Returns list of ActualResult (excludes failed fetches).
    """
    client = _get_client()
    log_path = LOGS_DIR / f"validator_{date}.log"
    actuals: list[ActualResult] = []

    with open(log_path, "w") as log_fh:
        log_fh.write(f"Validator run for {date} — {len(matches)} matches\n")
        logger.info("Fetching results for %d matches on %s", len(matches), date)

        for query in matches:
            result = _fetch_single_result(client, query, log_fh)
            if result is not None:
                actuals.append(result)
                logger.info(
                    "%s vs %s → %d-%d (%s) [%s]",
                    result.home_team, result.away_team,
                    result.home_score, result.away_score,
                    result.result, result.confidence,
                )

    # Save actuals
    out_path = ACTUALS_DIR / f"{date}_actuals.json"
    with open(out_path, "w") as f:
        json.dump([r.model_dump() for r in actuals], f, indent=2)
    logger.info("Saved %d actuals → %s", len(actuals), out_path)

    return actuals
