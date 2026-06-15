"""
Railway build script — runs full training pipeline.
Executed once at build time; outputs are baked into the deployment image.

Required env vars:
  KAGGLE_USERNAME  — your Kaggle username
  KAGGLE_KEY       — your Kaggle API key
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build")

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str) -> None:
    logger.info("=" * 60)
    logger.info("Running: %s", script)
    logger.info("=" * 60)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        logger.error("FAILED: %s (exit %d)", script, result.returncode)
        sys.exit(result.returncode)
    logger.info("OK: %s", script)


if __name__ == "__main__":
    run("phase1_data_foundation.py")
    run("phase2_feature_engineering.py")
    run("phase3_model_training.py")
    logger.info("Build complete — models saved to models/")
