"""Schema and quality validation for the football dataset."""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team",
    "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Validation {status} | errors={len(self.errors)} warnings={len(self.warnings)}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run all validation checks. Raises ValueError if any error found."""
    report = ValidationReport()

    _check_columns(df, report)
    _check_dtypes(df, report)
    _check_no_null_scores(df, report)
    _check_score_range(df, report)
    _check_date_range(df, report)
    _check_result_column(df, report)
    _check_no_self_match(df, report)

    logger.info(report.summary())
    if not report.passed:
        raise ValueError(f"Dataset validation failed:\n{report.summary()}")
    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.fail(f"Missing columns: {missing}")


def _check_dtypes(df: pd.DataFrame, report: ValidationReport) -> None:
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        report.fail("'date' column is not datetime dtype")
    for col in ["home_score", "away_score"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report.fail(f"'{col}' is not numeric dtype")


def _check_no_null_scores(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in ["home_score", "away_score"]:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                report.fail(f"'{col}' has {n} null values")


def _check_score_range(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in ["home_score", "away_score"]:
        if col not in df.columns:
            continue
        if (df[col] < 0).any():
            report.fail(f"'{col}' has negative values")
        if (df[col] > 30).any():
            n = (df[col] > 30).sum()
            report.warn(f"'{col}' has {n} values > 30 (possible data error)")


def _check_date_range(df: pd.DataFrame, report: ValidationReport) -> None:
    if "date" not in df.columns:
        return
    min_date = df["date"].min()
    max_date = df["date"].max()
    logger.info("Date range: %s → %s", min_date.date(), max_date.date())
    if min_date < pd.Timestamp("1872-01-01"):
        report.warn(f"Earliest date {min_date.date()} predates first international match (1872)")


def _check_result_column(df: pd.DataFrame, report: ValidationReport) -> None:
    if "result" not in df.columns:
        report.warn("'result' column not present — run cleaner first")
        return
    invalid = ~df["result"].isin([0, 1, 2])
    if invalid.any():
        report.fail(f"'result' column has {invalid.sum()} values outside {{0,1,2}}")


def _check_no_self_match(df: pd.DataFrame, report: ValidationReport) -> None:
    if "home_team" in df.columns and "away_team" in df.columns:
        self_matches = (df["home_team"] == df["away_team"]).sum()
        if self_matches > 0:
            report.fail(f"{self_matches} rows where home_team == away_team")
