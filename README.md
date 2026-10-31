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

## Mondrian split conformal (group-conditional regression)

When coverage must hold *inside* each segment (region, device, time bucket),
use Mondrian split conformal. Residuals are calibrated per group so a rare /
hard group gets its own wider quantile instead of borrowing the easy groups'
tight threshold:

```python
from conformal_kit import MondrianSplitConformalRegressor

conformal = MondrianSplitConformalRegressor(alpha=0.1).fit(
    y_calibration, calibration_pred, groups=calibration_region
)
lower, upper = conformal.predict_interval(test_pred, groups=test_region)
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

## Full conformal ridge regression (exact, no calibration split)

Jackknife+ pays for not holding data out with a `1 - 2 * alpha` guarantee. *Full* conformal prediction keeps the full `1 - alpha` guarantee and still uses every point for training, at the cost of refitting the model for every candidate label. For ridge regression that refit is linear in the candidate label `z`, so the exact prediction set can be computed in closed form: the Ridge Regression Confidence Machine (Nouretdinov, Melluish & Vovk, ICML 2001; Vovk, Gammerman & Shafer, 2005). Each training point's residual `a_i + b_i z` is compared with the test point's, the roots of `|a_i + b_i z| = |a_t + b_t z|` are swept in order, and the set `{z : p(z) > alpha}` is returned exactly, with no grid. A Sherman-Morrison update keeps each test point at `O(n p)`.

```python
from conformal_kit import FullConformalRidgeRegressor

model = FullConformalRidgeRegressor(alpha=0.1, ridge=1.0).fit(X_train, y_train)
lower, upper = model.predict_interval(X_test)   # convex hull of each exact set
sets = model.predict_set(X_test)                 # list of (low, high) pieces per row
model.p_value(X_test[0], [0.0, 1.5])             # full-conformal p-values of candidate labels
```

From the CLI, with raw feature columns plus `y_true`:

```bash
conformal-kit full-ridge --train train.csv --test test.csv --ridge 1.0 --alpha 0.1 --out intervals.csv
```

Use it when the model is (or can be) a ridge regression on a small or medium sample and no calibration split can be spared. If `(n + 1) * alpha < 1` every label is kept and the interval is unbounded, as the guarantee requires.

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

## Clustered conformal: class-conditional coverage with few examples per class

Mondrian needs `ceil(1/alpha) - 1` calibration points *per class*. With many classes and only a few examples of each, its thresholds become noisy or impossible. Split conformal pools every class together. **Clustered conformal prediction** (Ding, Fannjiang, Jordan & Tibshirani, NeurIPS 2023) sits between the two:

1. Part of the calibration data is used to embed each class as the quantiles of its scores.
2. Classes are grouped with weighted k-means.
3. Each cluster gets its own conformal threshold. Rare classes fall back to the marginal threshold.

```python
from conformal_kit import ClusteredConformalClassifier, class_coverage_gap

cc = ClusteredConformalClassifier(alpha=0.1, method="lac", seed=0).fit(y_cal, p_cal)
sets = cc.predict_set(p_test)
cc.n_clusters_, cc.class_to_cluster_, cc.cluster_quantiles_
class_coverage_gap(y_test, sets, alpha=0.1)["cov_gap"]   # mean |coverage_k - 0.9|
```

`n_clusters` and `frac_clustering` default to the paper's heuristic (`clustering_parameters`). Coverage is exact within each cluster, so it also holds marginally.

On a synthetic 100-class problem with two kinds of class difficulty and 20 calibration points per class:

| Method | CovGap | Mean set size |
| --- | --- | --- |
| Split | 12.8% | 29.9 |
| Mondrian | 6.1% | 27.9 |
| Clustered | 5.1% | 24.5 |

With 50 points per class the CovGaps were 10.9% (split), 3.9% (Mondrian) and 2.5% (clustered).

The CLI exposes it as `conformal-kit classify --method clustered` (APS scores).

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

## Conformal PID control: streaming intervals for time series

`conformal_pid` and `ConformalPIDRegressor` implement conformal PID control
(Angelopoulos, Candès & Tibshirani, NeurIPS 2023). ACI adjusts the
miscoverage *level*. PID control instead steers the score *threshold* `q_t`
directly, using three terms:

* **P (quantile tracking)**: `q <- q + lr * (err_t - alpha)`. On *any*
  bounded score stream, long-run miscoverage is within
  `(B + lr) / (lr * T)` of `alpha`.
* **I (error integration)**: `K_I * tan(sum(err - alpha) * log t / (t * C_sat))`.
  It reacts quickly to a run of misses.
* **D (scorecasting)**: an optional forecast of the next score, e.g.
  `ewma_scorecaster(halflife)`, so that trend and seasonality are anticipated.

```python
from conformal_kit import ConformalPIDRegressor, conformal_pid, ewma_scorecaster

res = conformal_pid(abs_residuals, alpha=0.1, lr=0.1, k_i=1.0,
                    scorecaster=ewma_scorecaster(20))
res.miscoverage, res.rolling_miscoverage(100)

model = ConformalPIDRegressor(alpha=0.1, lr=0.05).fit(y_cal, pred_cal)
lower, upper = model.predict_update(pred_stream, y_stream)  # issued before each label
```

On the CLI, run `conformal-kit pid --stream stream.csv --lr 0.1 --k-i 1 --ewma-halflife 20 --warmup 100 --out pid.csv`.

## Weighted conformal: coverage under covariate shift

Split conformal assumes calibration and test data are exchangeable. Under
**covariate shift** the feature distribution moves (new region, new device mix,
a different patient population) while `P(y | x)` stays the same, and plain
split conformal quietly under-covers wherever the test data has drifted into
harder regions. Weighted split conformal (Tibshirani, Barber, Candes, Ramdas
2019) reweights each calibration score by the likelihood ratio
`w(x) = dP_test(x) / dP_calibration(x)` and takes a weighted quantile, which
restores the `1 - alpha` guarantee on the shifted distribution.

```python
from conformal_kit import (
    WeightedConformalRegressor,
    likelihood_ratio_from_probabilities,
)

# w(x) from a domain classifier trained on pooled features to predict
# "is this row from the test pool?" (any probabilistic classifier works).
w_cal = likelihood_ratio_from_probabilities(p_test_cal, n_cal, n_test, clip=0.01)
w_test = likelihood_ratio_from_probabilities(p_test_new, n_cal, n_test, clip=0.01)

model = WeightedConformalRegressor(alpha=0.1).fit(y_cal, pred_cal, w_cal)
lower, upper = model.predict_interval(pred_test, w_test)
model.effective_n_   # Kish effective sample size of the calibration weights
```

Each test point gets its own width, because its own weight enters the
quantile. A test point whose weight dwarfs the calibration mass gets an
**infinite** interval, which is the honest answer for extrapolation. With all
weights equal the method reduces exactly to split conformal.
`normalize=True` combines it with per-point difficulty scaling.
`weighted_conformal_quantile` and `effective_sample_size` are exposed for
custom score functions.

```bash
conformal-kit weighted --calibration cal.csv --test test.csv --alpha 0.1 --out intervals.csv
```

`cal.csv` holds `y_true,y_pred,weight` and `test.csv` holds `y_pred,weight`.
Use `--weight-column` to pick a different column name.

## Conformal risk control: bounding expected loss

Split conformal bounds the miscoverage probability. Conformal risk control
(Angelopoulos, Bates, Fisch, Lei & Schuster 2022) bounds the expected value of
any bounded loss that shrinks as a threshold `lambda` grows, picking

`lambda_hat = inf { lambda : (n R_n(lambda) + B) / (n + 1) <= alpha }`

so that `E[loss] <= alpha` at finite `n`. `MultilabelRiskController`
applies it to multilabel classification: the set `{k : p_k >= 1 - lambda}`
has expected false negative rate at most `alpha`. The search over `lambda`
is exact because FNR only changes at the true labels' scores.

```python
from conformal_kit import MultilabelRiskController, false_negative_rate

ctrl = MultilabelRiskController(alpha=0.1).fit(y_cal_multi_hot, p_cal)
sets = ctrl.predict_set(p_test)           # boolean (n, n_labels)
false_negative_rate(y_test_multi_hot, sets)  # about 0.1 or below
ctrl.threshold_                            # probability cut-off 1 - lambda_hat
```

For any other monotone loss, build a `(n, m)` loss matrix over increasing
candidate `lambdas` and call `conformal_risk_threshold(losses, lambdas, alpha)`.

## Learn then Test: risk control with high probability

Conformal risk control bounds the *expected* loss. Averaged over calibration sets that is exact, but roughly half of all calibration draws still give a threshold whose true risk is above `alpha`. Learn then Test (Angelopoulos, Bates, Candès, Jordan & Lei, 2021) gives the stronger **(alpha, delta)** guarantee `P(R(lambda_hat) <= alpha) >= 1 - delta`. Each candidate `lambda` is a hypothesis test of `R(lambda) > alpha`, scored with a concentration-inequality p-value, and a family-wise error procedure keeps the `lambda` values whose null is rejected. The loss does **not** have to be monotone in `lambda`.

```python
from conformal_kit import LTTMultilabelController, learn_then_test

# Multilabel sets with FNR <= 10% for at least 90% of calibration draws
ctrl = LTTMultilabelController(alpha=0.1, delta=0.1).fit(y_cal, probs_cal)
sets = ctrl.predict_set(probs_test)   # labels with p >= 1 - ctrl.lambda_

# Any loss matrix: losses[i, j] = loss of calibration point i at lambdas[j]
res = learn_then_test(losses, lambdas, alpha=0.1, delta=0.1,
                      p_value="hb",            # hb | hoeffding | bentkus | binomial (0/1 losses)
                      method="fixed_sequence") # fixed_sequence | bonferroni | holm
res.lambda_hat, res.valid, res.p_values
```

- `hoeffding_p_value`, `bentkus_p_value`, `hoeffding_bentkus_p_value` (the minimum of the two, the default) and the exact `binomial_p_value` for 0/1 losses.
- `fwer_reject(p, delta, method, order)` implements Bonferroni, Holm and fixed-sequence testing. With a monotone loss, fixed sequence from the safest `lambda` down gives RCPS-style control with no multiplicity penalty. For a non-monotone loss, Bonferroni or Holm certifies only the safe region.
- `learn_then_test` returns an `LTTResult` with the empirical risks, p-values, certified mask and `lambda_hat`, or `None` if nothing passes.

CLI: `conformal-kit ltt --losses losses.csv --alpha 0.1 --delta 0.1`. The header row holds the `lambda` values and each row is one calibration point. The command prints the certified `lambda`, the CRC threshold for comparison, and optionally a per-`lambda` CSV.

## Ensemble Batch Prediction Intervals (EnbPI)

Time series break the exchangeability assumption that split conformal, CQR and
jackknife+ rely on. **EnbPI** (Xu and Xie, 2021) keeps a sliding pool of
leave-one-bootstrap residuals from an ensemble and rebuilds the interval at
every step from the most recent residuals. After each outcome arrives,
`update` drops the oldest residual and appends the fresh one.

```python
from conformal_kit import EnbPIRegressor

class MeanModel:
    def fit(self, X, y):
        self.mean_ = float(y.mean())
        return self
    def predict(self, X):
        import numpy as np
        return np.full(len(X), self.mean_)

model = EnbPIRegressor(MeanModel(), alpha=0.1, n_estimators=20, max_resid=50, random_state=0)
model.fit(X_train, y_train)
for x_t, y_t in stream:
    lower, upper = model.predict_interval(x_t.reshape(1, -1))
    y_hat = float(model.predict(x_t.reshape(1, -1)))
    model.update(y_true=y_t, y_pred=y_hat)
```

The CLI mirrors the sequential construction on a CSV of `y_true,y_pred`:

```bash
conformal-kit enbpi --data series.csv --alpha 0.1 --train-size 0.5 --out intervals.csv
```

## Venn-Abers: calibrated probabilities with a validity guarantee

Prediction sets say which labels are plausible. **Inductive Venn-Abers
predictors** (Vovk & Petej, 2014) give a calibrated *probability* instead.
For each test score the calibration set is augmented once with label 0 and
once with label 1, and an isotonic regression is fitted to each. The two
fitted values form a pair `(p0, p1)` with `p0 <= p1`, and under
exchangeability one of the two is perfectly calibrated. The gap
`p1 - p0` is wide where calibration data are thin, so it doubles as an
uncertainty signal. `predict_proba` merges the pair into a single
probability, `p1 / (1 - p0 + p1)`, which is minimax-optimal for log loss.
Probability matrices with more than two classes are calibrated one-vs-rest
and then renormalized.

```python
from conformal_kit import VennAbersCalibrator, venn_abers_probabilities

cal = VennAbersCalibrator().fit(p_cal[:, 1], y_cal)   # binary: positive-class scores
p0, p1 = cal.predict_interval(p_test[:, 1])           # multiprobability pair
probs = cal.predict_proba(p_test[:, 1])               # (n, 2) merged probabilities
cal.interval_width(p_test[:, 1])                      # p1 - p0 per row

VennAbersCalibrator().fit(p_cal, y_cal).predict_proba(p_test)  # multiclass, one-vs-rest
```

## Conformal outlier detection with FDR control

Any anomaly score (an isolation-forest score, an autoencoder's reconstruction
error, a negative log-likelihood) becomes a valid p-value once it is compared
with the scores of held-out inliers:
`p(x) = (1 + #{s_i >= s(x)}) / (n + 1)`, where a larger score means more
anomalous. If `x` is an inlier, then `P(p(x) <= t) <= t`.
`ConformalOutlierDetector` provides two ways to use these p-values:

- `method="marginal"` flags `p <= alpha`, so each inlier is wrongly flagged
  with probability at most `alpha`.
- `method="bh"` (the default) applies Benjamini–Hochberg across the test
  batch. Conformal p-values that share one calibration set are PRDS, so the
  **false discovery rate** stays at or below `alpha` (Bates et al., *Annals
  of Statistics* 2023).

`randomized=True` breaks ties at random, which makes the p-values exactly
uniform for inliers. That helps when scores are discrete.

```python
from conformal_kit import ConformalOutlierDetector, conformal_p_values, benjamini_hochberg

det = ConformalOutlierDetector(alpha=0.1, method="bh").fit(inlier_scores_cal)
flags = det.predict(test_scores)                 # 1 = outlier
report = det.decision_report(test_scores)        # p_values, q_values, flagged
```

## Conformal e-values and e-BH (derandomised outlier detection)

`conformal_e_values` builds the BH-thresholded conformal e-values of Bashari
et al. (NeurIPS 2023); `e_bh` is the e-BH procedure (Wang & Ramdas 2022),
which controls FDR under arbitrary dependence. e-BH on a single split's
e-values rejects exactly what BH on conformal p-values rejects, and since
e-values can be averaged, `derandomized_e_values` combines several random
calibration splits into one stable, still FDR-valid decision.
`p_to_e` / `e_to_p` are the standard calibrators.

```python
from conformal_kit import conformal_e_values, derandomized_e_values, e_bh

runs = [conformal_e_values(cal_split, test_scores, alpha=0.1)[0] for cal_split in splits]
outliers = e_bh(derandomized_e_values(runs), alpha=0.1)
```

## Conformal selection: pick candidates with y > c at a target FDR

Screening problems (drug candidates, job applicants, loans) ask a different
question from prediction intervals: *which test units have an outcome above a
threshold `c`?* `conformal_selection` (Jin & Candès, "Selection by Prediction
with Conformal p-values", JMLR 2023) turns any regression model into a
selection rule whose **false discovery rate** — the share of selected units
that actually have `y <= c` — stays at or below `q`, in finite samples and
without assumptions on the model.

Each test unit gets the p-value
`p_j = (#{V_i < V(X_j, c)} + 1 + #{V_i = V(X_j, c)}) / (n + 1)` and BH is
applied at level `q`. The default `score="clipped"`
(`V(x, y) = M * 1{y > c} + c - mu(x)`) removes calibration units that are
already above `c` from the comparison, which is much more powerful than the
plain residual score `y - mu(x)` (`score="residual"`).

```python
from conformal_kit import ConformalSelector, conformal_selection, selection_report

selected, p_values, q_values = conformal_selection(y_cal, pred_cal, pred_test, threshold=1.0, q=0.1)
selection_report(selected, y_test, threshold=1.0)   # realised FDP and power, when labels exist

sel = ConformalSelector(threshold=1.0, q=0.1).fit(y_cal, pred_cal)
sel.decision_report(pred_test)                      # p_values, q_values, selected
```

Thresholds may also be per-unit (`threshold=c_cal, test_threshold=c_test`).

## Conformal predictive distributions (full predictive CDFs)

`SplitConformalPredictiveSystem` turns signed calibration residuals into a
full predictive distribution for every test point (Vovk et al., split
conformal predictive systems). The randomized CDF
`Q(y, tau) = (#{r_i < s} + tau * (#{r_i = s} + 1)) / (n + 1)` is *calibrated in
probability*: under exchangeability `Q(y_test, tau)` is exactly uniform, so
every quantile has the advertised frequency for any model.

```python
from conformal_kit import SplitConformalPredictiveSystem, pit_uniformity_report

cps = SplitConformalPredictiveSystem().fit(y_cal, pred_cal)
cps.quantile([0.1, 0.5, 0.9], pred_test)      # (n_test, 3), P(Y <= q_p) >= p
cps.predict_interval(pred_test, alpha=0.1)    # central 90% interval
cps.cdf(threshold, pred_test)                 # P(Y <= threshold) per point
cps.crps(y_test, pred_test).mean()            # proper score for model comparison
pit_uniformity_report(cps.pit(y_test, pred_test, seed=0))["ks_statistic"]
```

`normalize=True` with a per-point `difficulty` scales residuals for
heteroskedastic data (lower CRPS, still calibrated). Upper quantiles beyond
what the calibration set can resolve return `+inf`.

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
conformal-kit weighted --calibration cal.csv --test test.csv --alpha 0.1
conformal-kit predictive --calibration cal.csv --test test.csv --quantiles 0.1,0.5,0.9 --out q.csv
conformal-kit outliers --calibration inlier_scores.csv --test scores.csv --alpha 0.1 --method bh --out flags.csv
conformal-kit ltt --losses losses.csv --alpha 0.1 --delta 0.1 --method fixed_sequence --out ltt.csv
```

`select` reads `y_true,y_pred` calibration rows and `y_pred` test rows and selects rows with `y > --threshold` at FDR `--alpha`
(`conformal-kit select --calibration cal.csv --test test.csv --threshold 1.0 --alpha 0.1 --out selected.csv`;
add `--test-labels` to report the realised FDP and power when the test CSV also has `y_true`).

`outliers` reads a `score` column (change it with `--score-column`) and writes `p_value,q_value,outlier` for every test row.

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
