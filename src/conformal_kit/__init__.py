"""Distribution-free uncertainty quantification via conformal prediction."""

from .aci import (
    AdaptiveConformalClassifier,
    AdaptiveConformalRegressor,
    AdaptiveConformalUpdater,
    aci_update,
    adaptive_conformal_quantile,
)
from .aps import APSClassifier, RAPSClassifier, aps_scores, raps_scores
from .classification import (
    MondrianConformalClassifier,
    SplitConformalClassifier,
    mondrian_quantiles,
)
from .cqr import ConformalizedQuantileRegressor, cqr_scores
from .cross_conformal import (
    CrossConformalClassifier,
    cross_conformal_p_values,
    cross_conformal_sets,
)
from .enbpi import EnbPIRegressor, enbpi_interval, enbpi_quantile
from .evaluation import (
    coverage_by_group,
    interval_coverage_report,
    set_coverage_report,
)
from .jackknife import JackknifePlusRegressor, jackknife_plus_interval
from .regression import SplitConformalRegressor, conformal_quantile

__version__ = "0.1.0"

__all__ = [
    "APSClassifier",
    "AdaptiveConformalClassifier",
    "AdaptiveConformalRegressor",
    "AdaptiveConformalUpdater",
    "ConformalizedQuantileRegressor",
    "CrossConformalClassifier",
    "EnbPIRegressor",
    "JackknifePlusRegressor",
    "MondrianConformalClassifier",
    "RAPSClassifier",
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "__version__",
    "aci_update",
    "adaptive_conformal_quantile",
    "aps_scores",
    "conformal_quantile",
    "coverage_by_group",
    "cqr_scores",
    "enbpi_interval",
    "enbpi_quantile",
    "cross_conformal_p_values",
    "cross_conformal_sets",
    "interval_coverage_report",
    "jackknife_plus_interval",
    "mondrian_quantiles",
    "raps_scores",
    "set_coverage_report",
]
