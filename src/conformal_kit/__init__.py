"""Distribution-free uncertainty quantification via conformal prediction."""

from .regression import SplitConformalRegressor, conformal_quantile

__version__ = "0.1.0"

__all__ = [
    "SplitConformalRegressor",
    "conformal_quantile",
    "__version__",
]
