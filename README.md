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

Fits intervals on a heteroskedastic synthetic problem, compares standard against normalized conformal, and writes a report to `examples/output/`.

## What this does not do

- The coverage guarantee is **marginal**, averaged over test points. It does not promise coverage within a subgroup; `interval_coverage_report` accepts a `groups` argument so you can check that yourself.
- It assumes **exchangeability**. Under distribution shift or on time series with trend, the guarantee lapses.
- It wraps a model, it does not improve one. A weak model gets valid but wide intervals.

## License

MIT
