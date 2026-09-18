# conformal-prediction-kit

Distribution-free prediction intervals and prediction sets with finite-sample coverage guarantees.

A point prediction tells you what the model thinks. It does not tell you how much to trust it. Conformal prediction turns any fitted model — no retraining, no distributional assumption — into one that outputs an interval (regression) or a set of labels (classification) that contains the truth at least `1 - alpha` of the time.

The guarantee is the point: given exchangeable data, coverage holds for any model, any sample size, and any data distribution. A badly calibrated model still gets valid coverage; it just pays for it with wider intervals.

## Install

```bash
pip install -e ".[dev]"
```

## Regression: prediction intervals

```python
import numpy as np
from conformal_kit import SplitConformalRegressor

# Predictions from any model, on a calibration set held out from training.
calibration_pred = model.predict(X_calibration)
conformal = SplitConformalRegressor(alpha=0.1).fit(y_calibration, calibration_pred)

lower, upper = conformal.predict_interval(model.predict(X_test))
```

`alpha=0.1` asks for 90% coverage. The interval width is one number — the calibrated quantile of absolute residuals — so every test point gets the same width.

For heteroskedastic data, where uncertainty varies across the input space, pass per-point difficulty estimates and the width scales with them:

```python
conformal = SplitConformalRegressor(alpha=0.1, normalize=True).fit(
    y_calibration, calibration_pred, difficulty=calibration_sigma
)
lower, upper = conformal.predict_interval(test_pred, difficulty=test_sigma)
```

## CQR: conformalized quantile regression

When the model already predicts a lower and an upper quantile, conformalize those bounds instead of a point prediction. CQR (Romano, Patterson, Candès 2019) measures how far calibration labels fall outside the predicted quantile interval and expands both sides by a finite-sample quantile of that score. Marginal coverage is `1 - alpha`; the width still follows the quantile model, so intervals stay narrow where the noise is small.

sklearn is not required: train any quantile regressor (typically at `alpha / 2` and `1 - alpha / 2`) and pass its predictions.

```python
from conformal_kit import ConformalizedQuantileRegressor

conformal = ConformalizedQuantileRegressor(alpha=0.1).fit(
    y_calibration, calibration_lower, calibration_upper
)
lower, upper = conformal.predict_interval(test_lower, test_upper)
```

If the two quantile bounds are identical, CQR reduces to split conformal around a point prediction.

## Jackknife+ and CV+: intervals without a calibration split

Split conformal is cheapest when you can spare a held-out calibration set. Jackknife+ spends that data on training instead, by refitting leave-one-out so every residual comes from a model that never saw that point. CV+ is the same idea with `K` folds rather than `n`. sklearn is not required: pass a duck-typed `fit` / `predict` estimator, or a callable that fits and returns a predictor.

```python
from conformal_kit import JackknifePlusRegressor

def mean_trainer(X_train, y_train):
    mu = float(y_train.mean())
    return lambda X: np.full(len(np.asarray(X)), mu)

# Leave-one-out jackknife+: n refits, every point trains a model.
conformal = JackknifePlusRegressor(alpha=0.1).fit(X, y, mean_trainer)
lower, upper = conformal.predict_interval(X_test)

# CV+ with 10 folds: the practical default when n refits are too slow.
conformal = JackknifePlusRegressor(alpha=0.1, n_splits=10).fit(X, y, mean_trainer)
```

### When to use which

- **Split conformal** when a fitted model and a held-out calibration set already exist, or when `n` is large enough that holding out 20–50% is cheap. One fit, `1 - alpha` coverage, constant (or difficulty-scaled) width.
- **CQR** when a quantile-regression model already produces lower and upper bounds. Same wrap-predictions API as split conformal; the interval width follows the quantile model instead of a separate difficulty estimate.
- **Jackknife+** when the sample is too small to spare a calibration split and `n` refits are affordable (linear models, small trees).
- **CV+** (`n_splits=10`) as the jackknife-style default for anything slower to fit: almost the same intervals, `K` refits.

The finite-sample guarantee for jackknife+ / CV+ is `1 - 2 * alpha`, not `1 - alpha`. In practice the intervals usually land close to the split-conformal target; the extra `alpha` is the price of not holding data out. Split conformal is the right default whenever a calibration split is affordable.

## Classification: prediction sets

```python
from conformal_kit import SplitConformalClassifier

conformal = SplitConformalClassifier(alpha=0.1, method="aps").fit(
    y_calibration, calibration_probabilities
)
sets = conformal.predict_set(test_probabilities)
```

Two scoring rules:

- `lac` (least ambiguous classifier) gives the smallest average set size, but coverage is uneven across classes.
- `aps` (adaptive prediction sets) gives larger sets that adapt to how uncertain each point is, with coverage spread more evenly.

## Evaluating coverage

Coverage is a claim to be checked, not assumed:

```python
from conformal_kit import interval_coverage_report

report = interval_coverage_report(y_test, lower, upper, alpha=0.1)
report["coverage"]          # empirical share of points inside the interval
report["within_tolerance"]  # whether that is consistent with the target
```

## CLI

```bash
conformal-kit calibrate --calibration cal.csv --test test.csv --alpha 0.1
conformal-kit evaluate --predictions intervals.csv --alpha 0.1
```

## Demo

```bash
python examples/run_demo.py
```

Fits intervals on a heteroskedastic synthetic problem, compares standard, normalized, and CQR conformal, and writes a report to `examples/output/`.

## What this does not do

- The coverage guarantee is **marginal**, averaged over test points. It does not promise coverage within a subgroup; `interval_coverage_report` accepts a `groups` argument so you can check that yourself.
- It assumes **exchangeability**. Under distribution shift or on time series with trend, the guarantee lapses.
- It wraps a model, it does not improve one. A weak model gets valid but wide intervals.

## License

MIT
