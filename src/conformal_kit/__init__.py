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
from .clustered import (
    ClusteredConformalClassifier,
    class_coverage_gap,
    clustering_parameters,
    weighted_kmeans,
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
from .full_conformal import FullConformalRidgeRegressor, ridge_conformal_set
from .jackknife import JackknifePlusRegressor, jackknife_plus_interval
from .ltt import (
    LTTMultilabelController,
    LTTResult,
    bentkus_p_value,
    binomial_p_value,
    fwer_reject,
    hoeffding_bentkus_p_value,
    hoeffding_p_value,
    learn_then_test,
    ltt_p_values,
)
from .outliers import ConformalOutlierDetector, benjamini_hochberg, conformal_p_values
from .predictive import (
    SplitConformalPredictiveSystem,
    crps_empirical,
    pit_uniformity_report,
)
from .regression import (
    MondrianSplitConformalRegressor,
    SplitConformalRegressor,
    conformal_quantile,
    mondrian_regression_quantiles,
)
from .risk_control import (
    MultilabelRiskController,
    conformal_risk_threshold,
    false_negative_rate,
)
from .venn_abers import VennAbersCalibrator, venn_abers_probabilities
from .weighted import (
    WeightedConformalRegressor,
    effective_sample_size,
    likelihood_ratio_from_probabilities,
    weighted_conformal_quantile,
)

__version__ = "0.1.0"

__all__ = [
    "ClusteredConformalClassifier",
    "class_coverage_gap",
    "clustering_parameters",
    "weighted_kmeans",
    "APSClassifier",
    "AdaptiveConformalClassifier",
    "AdaptiveConformalRegressor",
    "AdaptiveConformalUpdater",
    "ConformalOutlierDetector",
    "ConformalizedQuantileRegressor",
    "CrossConformalClassifier",
    "EnbPIRegressor",
    "FullConformalRidgeRegressor",
    "JackknifePlusRegressor",
    "LTTMultilabelController",
    "LTTResult",
    "MondrianConformalClassifier",
    "MondrianSplitConformalRegressor",
    "MultilabelRiskController",
    "RAPSClassifier",
    "SplitConformalClassifier",
    "SplitConformalPredictiveSystem",
    "SplitConformalRegressor",
    "VennAbersCalibrator",
    "WeightedConformalRegressor",
    "__version__",
    "aci_update",
    "adaptive_conformal_quantile",
    "aps_scores",
    "benjamini_hochberg",
    "bentkus_p_value",
    "binomial_p_value",
    "conformal_p_values",
    "ridge_conformal_set",
    "conformal_quantile",
    "conformal_risk_threshold",
    "coverage_by_group",
    "cqr_scores",
    "cross_conformal_p_values",
    "cross_conformal_sets",
    "crps_empirical",
    "effective_sample_size",
    "enbpi_interval",
    "enbpi_quantile",
    "false_negative_rate",
    "fwer_reject",
    "hoeffding_bentkus_p_value",
    "hoeffding_p_value",
    "interval_coverage_report",
    "jackknife_plus_interval",
    "learn_then_test",
    "likelihood_ratio_from_probabilities",
    "ltt_p_values",
    "mondrian_quantiles",
    "mondrian_regression_quantiles",
    "pit_uniformity_report",
    "raps_scores",
    "set_coverage_report",
    "venn_abers_probabilities",
    "weighted_conformal_quantile",
]
