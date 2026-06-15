"""Phase 1: Data Foundation — load, clean, validate, split, save."""

import logging
import sys
from pathlib import Path

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from src.data.cleaner import clean, split_by_date
from src.data.loader import load_raw
from src.data.validator import validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    logger.info("=== Step 1: Load ===")
    raw = load_raw()
    logger.info("Raw shape: %s | columns: %s", raw.shape, list(raw.columns))
    logger.info("Sample:\n%s", raw.head(3).to_string())

    # 2. Clean
    logger.info("=== Step 2: Clean ===")
    cleaned = clean(raw)

    # 3. Validate
    logger.info("=== Step 3: Validate ===")
    report = validate(cleaned)
    logger.info(report.summary())

    # 4. Split
    logger.info("=== Step 4: Split ===")
    train, val, test = split_by_date(cleaned)

    # 5. Save
    logger.info("=== Step 5: Save ===")
    out = PROCESSED_DIR / "matches_clean.parquet"
    cleaned.to_parquet(out, index=False)
    logger.info("Saved full cleaned dataset → %s (%d rows)", out, len(cleaned))

    # Save splits too
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"matches_{name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Saved %s split → %s (%d rows)", name, path, len(split_df))

    # 6. Summary
    logger.info("=== Phase 1 Complete ===")
    _print_summary(cleaned, train, val, test)


def _print_summary(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Total matches:     {len(df):>7,}")
    print(f"Train (<2018):     {len(train):>7,}  ({len(train)/len(df)*100:.1f}%)")
    print(f"Val (2018–2022):   {len(val):>7,}  ({len(val)/len(df)*100:.1f}%)")
    print(f"Test (>2022):      {len(test):>7,}  ({len(test)/len(df)*100:.1f}%)")
    print(f"Date range:        {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Unique home teams: {df['home_team'].nunique()}")
    print(f"Unique away teams: {df['away_team'].nunique()}")
    print(f"Tournaments:       {df['tournament'].nunique()}")
    print()
    print("Result distribution:")
    result_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
    vc = df["result"].value_counts().sort_index(ascending=False)
    for code, count in vc.items():
        print(f"  {result_map[code]:10s}  {count:>6,}  ({count/len(df)*100:.1f}%)")
    print()
    print("Top 10 tournaments:")
    print(df["tournament"].value_counts().head(10).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
