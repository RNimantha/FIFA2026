"""Load and cache the international football results dataset from Kaggle."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_HANDLE = "martj42/international-football-results-from-1872-to-2017"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
CACHE_PATH = RAW_DIR / "matches_raw.parquet"

# The main results file in the Kaggle dataset
RESULTS_FILENAME = "results.csv"


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Return raw dataset. Downloads from Kaggle on first call, then uses cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not force_download:
        logger.info("Loading from cache: %s", CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)

    logger.info("Downloading dataset from Kaggle: %s", DATASET_HANDLE)
    try:
        import kagglehub

        dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
        dataset_path = Path(dataset_path)
        logger.info("Downloaded to: %s", dataset_path)

        # Find results.csv (could be in a subdirectory)
        csv_files = list(dataset_path.rglob(RESULTS_FILENAME))
        if not csv_files:
            all_csvs = list(dataset_path.rglob("*.csv"))
            logger.info("Available CSVs: %s", all_csvs)
            raise FileNotFoundError(
                f"'{RESULTS_FILENAME}' not found in {dataset_path}. "
                f"Available: {[f.name for f in all_csvs]}"
            )

        results_csv = csv_files[0]
        logger.info("Reading: %s", results_csv)
        df = pd.read_csv(results_csv)

    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed: {e}\n"
            "Ensure kagglehub is installed and Kaggle API credentials are configured."
        ) from e

    logger.info("Downloaded %d rows, %d cols. Caching to %s", *df.shape, CACHE_PATH)
    df.to_parquet(CACHE_PATH, index=False)
    return df
