"""Distribution-free uncertainty quantification via conformal prediction."""

from .classification import SplitConformalClassifier
from .evaluation import (
    coverage_by_group,
    interval_coverage_report,
    set_coverage_report,
)
from .jackknife import JackknifePlusRegressor, jackknife_plus_interval
from .regression import SplitConformalRegressor, conformal_quantile

__version__ = "0.1.0"

__all__ = [
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "JackknifePlusRegressor",
    "conformal_quantile",
    "jackknife_plus_interval",
    "coverage_by_group",
    "interval_coverage_report",
    "set_coverage_report",
    "__version__",
]
