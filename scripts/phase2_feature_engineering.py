"""Phase 2: Feature Engineering — build full feature matrix with zero data leakage."""

import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from src.features.engineer import FEATURES, build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"


def main() -> None:
    # 1. Load cleaned data
    logger.info("=== Step 1: Load cleaned data ===")
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    logger.info("Loaded %d rows", len(df))

    # 2. Build features
    logger.info("=== Step 2: Build features ===")
    t0 = perf_counter()
    features_df = build_features(df)
    elapsed = perf_counter() - t0
    logger.info("Feature engineering took %.1fs", elapsed)

    # 3. Save
    logger.info("=== Step 3: Save ===")
    out = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out, index=False)
    logger.info("Saved → %s", out)

    # 4. Also save split feature files (for model training in Phase 3)
    for split_name in ["train", "val", "test"]:
        split_path = PROCESSED_DIR / f"matches_{split_name}.parquet"
        if not split_path.exists():
            logger.warning("Split file not found: %s (run phase1 first)", split_path)
            continue
        split_dates = pd.read_parquet(split_path, columns=["date"])["date"]
        split_mask = features_df["date"].isin(split_dates)
        split_features = features_df[split_mask]
        out_path = PROCESSED_DIR / f"features_{split_name}.parquet"
        split_features.to_parquet(out_path, index=False)
        logger.info("Saved %s features → %s (%d rows)", split_name, out_path, len(split_features))

    # 5. Summary
    _print_summary(features_df)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"Total rows:        {len(df):>7,}")
    print(f"Feature columns:   {len(FEATURES):>7}")
    print(f"NaN count:         {df[FEATURES].isna().sum().sum():>7}")
    print()
    print("Feature statistics:")
    print(df[FEATURES].describe().round(3).to_string())
    print()
    print("ELO sanity check:")
    print(f"  home_elo  mean={df['home_elo'].mean():.0f}  min={df['home_elo'].min():.0f}  max={df['home_elo'].max():.0f}")
    print(f"  away_elo  mean={df['away_elo'].mean():.0f}  min={df['away_elo'].min():.0f}  max={df['away_elo'].max():.0f}")
    print()
    print("Form sanity check:")
    print(f"  home_form5_ppg   mean={df['home_form5_ppg'].mean():.3f}")
    print(f"  away_form5_ppg   mean={df['away_form5_ppg'].mean():.3f}")
    print()
    print("H2H sanity check:")
    print(f"  h2h_total_matches  mean={df['h2h_total_matches'].mean():.1f}  max={df['h2h_total_matches'].max()}")
    print(f"  h2h_home_win_rate  mean={df['h2h_home_win_rate'].mean():.3f}")
    print()
    print("Tournament weights:")
    print(df.groupby("tournament")["tournament_weight"].first().sort_values(ascending=False).head(10).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
