"""Distribution-free uncertainty quantification via conformal prediction."""

from .aci import (
    AdaptiveConformalClassifier,
    AdaptiveConformalRegressor,
    AdaptiveConformalUpdater,
    aci_update,
    adaptive_conformal_quantile,
)
from .classification import (
    MondrianConformalClassifier,
    SplitConformalClassifier,
    mondrian_quantiles,
)
from .cqr import ConformalizedQuantileRegressor, cqr_scores
from .evaluation import (
    coverage_by_group,
    interval_coverage_report,
    set_coverage_report,
)
from .jackknife import JackknifePlusRegressor, jackknife_plus_interval
from .regression import SplitConformalRegressor, conformal_quantile

__version__ = "0.1.0"

__all__ = [
    "AdaptiveConformalClassifier",
    "AdaptiveConformalRegressor",
    "AdaptiveConformalUpdater",
    "ConformalizedQuantileRegressor",
    "JackknifePlusRegressor",
    "MondrianConformalClassifier",
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "__version__",
    "aci_update",
    "adaptive_conformal_quantile",
    "conformal_quantile",
    "coverage_by_group",
    "cqr_scores",
    "interval_coverage_report",
    "jackknife_plus_interval",
    "mondrian_quantiles",
    "set_coverage_report",
]
