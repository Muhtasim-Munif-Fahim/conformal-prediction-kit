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
- **ACI** when points arrive as a stream and exchangeability may fail — a distribution shift, a time series with drift. The miscoverage level moves after every outcome so long-run coverage tracks `1 - alpha`.

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

Both give *marginal* coverage: the average over all test points is at least `1 - alpha`. A rare or hard class can still be under-covered while easy classes make up the average.

`method="aps"` is the inductive split form of adaptive prediction sets (Romano, Sesia, Candès 2020): calibrate the cumulative-probability score on a held-out set, then return the label set at the target coverage. The probabilities have to come from a model that did not train on those calibration rows.

## APS and regularized APS (RAPS)

`APSClassifier` is that same split-conformal adaptive prediction set under its own name. The nonconformity of a label is the cumulative softmax (or probability) mass swept up in descending order until the label is included. `predict_set` and `predict` return the same boolean mask: a class is in the set when its score is at most the calibrated quantile.

`RAPSClassifier` is regularized APS (Angelopoulos, Bates, Malik, Jordan 2021). The score gains a penalty once a label's rank passes `k_reg`:

```text
score(y) = cumulative mass through y + penalty * max(rank(y) - k_reg, 0)
```

`rank` starts at 1 for the most probable class. `penalty=0`, or a `k_reg` at least as large as the number of classes, is exactly APS. A positive penalty drops classes that only just cleared the threshold, which is how RAPS shortens sets when the softmax is diffuse. The quantile is computed from the penalized calibration scores, so marginal coverage stays at least `1 - alpha` under exchangeability. Sets stay reproducible; the randomized last-class draw that would land closer to the target is the same one split APS leaves out.

```python
from conformal_kit import APSClassifier, RAPSClassifier, set_coverage_report

aps = APSClassifier(alpha=0.1).fit(y_calibration, calibration_probabilities)
raps = RAPSClassifier(alpha=0.1, penalty=0.01, k_reg=1).fit(
    y_calibration, calibration_probabilities
)
mask = raps.predict(test_probabilities)
report = set_coverage_report(y_test, mask, alpha=0.1)
```

`penalty` and `k_reg` change set size, not the guarantee. Leave `k_reg=1` to penalize everything past the top class. Raise `penalty` when the sets are still longer than you can use. The same mask works with `set_coverage_report`.

From the command line, pass one column per class probability:

```bash
conformal-kit classify --calibration cal.csv --test test.csv --alpha 0.1 --method raps
```

`cal.csv` needs `y_true,p0,p1,...`. `test.csv` needs `p0,p1,...`, and may also carry `y_true` — when it does, the command prints a coverage report and exits nonzero if that coverage misses the target. `--method aps` is the unregularized score. `--penalty` and `--k-reg` apply only to `raps` (defaults `0.01` and `1`).

## Mondrian: class-conditional prediction sets

Mondrian conformal prediction (Vovk, Lindsay, Nouretdinov, Gammerman) gives each label its own threshold. Calibration scores are grouped by the true class and the finite-sample quantile is taken inside each group, so coverage holds *given the true label*, not only on average.

```python
from conformal_kit import MondrianConformalClassifier

conformal = MondrianConformalClassifier(alpha=0.1, method="aps").fit(
    y_calibration, calibration_probabilities
)
sets = conformal.predict_set(test_probabilities)
```

Same `fit` / `predict_set` / `predict_labels` / `set_sizes` API as `SplitConformalClassifier`, and the same `lac` / `aps` scoring rules. The difference is one number versus one number per class.

The cost is data. Every class needs enough calibration examples to support the finite-sample quantile — at `alpha=0.1` that is 9 points *per class*, not 9 points overall. A class that never appears cannot get a threshold; `fit` refuses rather than invent one.

Use split conformal when you only need average coverage. Use Mondrian when a missed class is as bad as a missed point — a rare diagnosis, a safety label, any setting where the class you fail on is the one that matters. Check it with `set_coverage_report(..., groups=y_test)`.

## Cross-conformal and CV+: classification sets without a calibration split

Split APS and LAC already cover the inductive case, where a fitted model and a held-out probability matrix are enough. Cross-conformal prediction (Vovk 2015) is the route that does not hold data out. Each training point is scored by a model that never saw it, and a candidate label is kept when its p-value clears `alpha`. `n_splits=None` is leave-one-out; `n_splits=K` is K-fold CV+, the same split of labour as jackknife+ versus CV+ for regression.

Regularized APS (RAPS) is a split-conformal score, on `RAPSClassifier`, not a cross-validation one. This section stays the route that does not hold data out.

```python
from conformal_kit import CrossConformalClassifier

def frequency_trainer(X_train, y_train):
    labels = np.asarray(y_train).ravel()
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    proba = counts / counts.sum()

    def predict_proba(X):
        return np.tile(proba, (len(np.asarray(X)), 1))

    return predict_proba

# Leave-one-out cross-conformal: n refits, every point trains a model.
conformal = CrossConformalClassifier(alpha=0.1, method="aps").fit(X, y, frequency_trainer)
sets = conformal.predict_set(X_test)

# K-fold CV+: the practical default when n refits are too slow.
conformal = CrossConformalClassifier(alpha=0.1, method="aps", n_splits=10).fit(
    X, y, frequency_trainer
)
```

The model is a duck-typed estimator with `fit` and `predict_proba`, or a callable that fits and returns `predict_proba`. sklearn is not required. `predict_proba` must return one column per class index `0 .. max(y)`, including on a fold that never saw some class — give the estimator the full class count when a fold can miss a label.

The p-value for candidate label `y` is

```text
p(y) = (1 + #{i : R_i >= s_{k(i)}(x, y)}) / (n + 1)
```

`R_i` is the nonconformity of training point `i` (`lac` or `aps`) under the model that held its fold out, and `s_{k(i)}(x, y)` is that same model's score for `y` at the test point. Ties count toward inclusion. An empty row keeps its most probable class, averaged across the fold models, so the sets stay reproducible. Randomizing ties would shrink them slightly and make them depend on a draw.

The finite-sample guarantee is the jackknife+ one: at least `1 - 2 * alpha`, up to a term that vanishes like `1 / sqrt(n)` and is zero for leave-one-out. In practice the sets usually land close to the split-conformal target. The extra `alpha` is the price of not holding data out. Split conformal is still the right default when a calibration split is affordable. `lac` still tends toward smaller sets; `aps` still adapts to each point. Because each fold model votes separately, a cross-conformal set need not be a prefix of a single probability ranking.

## ACI: coverage under distribution shift

Split conformal's guarantee needs exchangeable calibration and test points. After a shift the frozen quantile is wrong, and coverage drops until you recalibrate. Adaptive conformal inference (Gibbs and Candès 2021) keeps a time-varying miscoverage level `alpha_t` and updates it after every outcome:

```text
alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
```

`err_t` is 1 if the interval (or set) missed, 0 if it covered. A miss lowers `alpha_t`, so the next finite-sample quantile of the residual window is more conservative; a hit raises it. The average of `err_t` tracks `alpha` at rate `O(1 / (gamma T))`, with no exchangeability assumption.

The wrap-predictions API matches split conformal. `fit` seeds the residual window at the target `alpha`. `predict_update` is the online loop: issue an interval, observe the label, adapt before the next point.

```python
from conformal_kit import AdaptiveConformalRegressor

conformal = AdaptiveConformalRegressor(alpha=0.1, gamma=0.05).fit(
    y_calibration, calibration_pred
)
lower, upper = conformal.predict_update(stream_pred, stream_y)
```

`gamma` is the step size. Smaller values track slowly and oscillate less; larger values react faster after a shift. `window_size` optionally forgets the oldest residuals; the default expanding window still works because `alpha_t` compensates.

The same updater wraps classification scores (`lac` or `aps`) as `AdaptiveConformalClassifier`, with `predict_set` / `predict_update` in place of intervals. Either way the object is a residual conformalizer: it never sees features and never retrains the model.

ACI restores *long-run* coverage along the stream. It does not restore a finite-sample exchangeability guarantee on any one window, and it does not give class-conditional coverage. Right after a sudden shift the next few sets can miss while `alpha_t` catches up.

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
conformal-kit classify --calibration cal.csv --test test.csv --alpha 0.1 --method raps
conformal-kit evaluate --predictions intervals.csv --alpha 0.1
```

`calibrate` reads `y_true,y_pred` and writes regression intervals. `classify` reads `y_true,p0,p1,...` and writes a prediction set per row (`--method aps` or `--method raps`). `evaluate` scores an interval file. The two input layouts are not interchangeable.

## Demo

```bash
python examples/run_demo.py
```

Fits intervals on a heteroskedastic synthetic problem, compares standard, normalized, and CQR conformal, and writes a report to `examples/output/`.

## What this does not do

- Split conformal's coverage guarantee is **marginal**, averaged over test points. It does not promise coverage within a subgroup. Mondrian conformal is the exception for *class labels*: each class gets its own threshold, so coverage holds conditionally on the true label. Other subgroups (age, region, ...) are still not guaranteed; `interval_coverage_report` and `set_coverage_report` accept a `groups` argument so you can check that yourself.
- Cross-conformal classification, like jackknife+ and CV+, guarantees about `1 - 2 * alpha` rather than `1 - alpha`. The gap is the price of training on every point. Split conformal remains exact at `1 - alpha` when a calibration set is available.
- Split conformal, CQR, jackknife+ and Mondrian assume **exchangeability**. Under distribution shift or on a time series with trend, that guarantee lapses. ACI is the exception for *long-run* coverage: `alpha_t` moves after every miss or hit so the average along the stream tracks `1 - alpha`, without assuming exchangeability. It does not restore conditional coverage, and a single window right after a shift can still under-cover.
- It wraps a model, it does not improve one. A weak model gets valid but wide intervals.

## License

MIT
