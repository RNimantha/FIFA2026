"""Clean and standardize the raw football dataset."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Cutoff: exclude very old data for main model (sparse & unreliable)
DATE_CUTOFF = "1950-01-01"

# Temporal split boundaries (see CLAUDE.md Rule 2)
TRAIN_END = "2018-01-01"
VAL_END = "2022-11-01"

TEAM_NAME_MAP: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "Czech Republic": "Czechia",
    "Congo DR": "DR Congo",
    "Trinidad and Tobago": "Trinidad & Tobago",
}

EXPECTED_COLUMNS = {
    "date", "home_team", "away_team",
    "home_score", "away_score",
    "tournament", "city", "country", "neutral",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns cleaned DataFrame.
    Logs shape before/after every transformation.
    """
    logger.info("START clean | shape=%s", df.shape)

    df = _cast_types(df)
    df = _drop_missing_scores(df)
    df = _filter_date(df)
    df = _standardize_team_names(df)
    df = _add_result_column(df)
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("END clean | shape=%s", df.shape)
    return df


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) splits based on fixed date boundaries."""
    train = df[df["date"] < TRAIN_END].copy()
    val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
    test = df[df["date"] >= VAL_END].copy()

    logger.info(
        "Split | train=%d  val=%d  test=%d  (total=%d)",
        len(train), len(val), len(test), len(df),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)
    logger.info("_cast_types | %s → %s", before, df.shape)
    return df


def _drop_missing_scores(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    logger.info("_drop_missing_scores | %d → %d rows", before, len(df))
    return df


def _filter_date(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["date"] >= DATE_CUTOFF].copy()
    logger.info("_filter_date (>=%s) | %d → %d rows", DATE_CUTOFF, before, len(df))
    return df


def _standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
    replaced = sum(1 for v in TEAM_NAME_MAP if v in df["home_team"].values or v in df["away_team"].values)
    logger.info("_standardize_team_names | %d team name variants remain unmapped", replaced)
    return df


def _add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'result' column: 2=Home Win, 1=Draw, 0=Away Win."""
    df = df.copy()
    df["result"] = (
        (df["home_score"] > df["away_score"]).astype(int) * 2
        + (df["home_score"] == df["away_score"]).astype(int)
    )
    return df
