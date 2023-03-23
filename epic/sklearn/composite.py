import numpy as np
import pandas as pd

from typing import Literal
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from sklearn.exceptions import NotFittedError
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .metrics import confusion_score
from .general import Classifier, LinearClassifier


class ThresholdCompositeClassifier(BaseEstimator, ClassifierMixin):
    """
    Compute a threshold to apply to probabilities based on precision or recall on test data.

    This classifier wraps a binary classifier.
    Upon training, the data is split into a training set and a testing set. The classifier is
    fitted on the training set, and the precision and recall curves are calculated on the
    testing set. A threshold is then determined based on the desired value of one of those statistics.
    Upon prediction, the threshold is applied to the predicted probabilities to determine the
    predicted class.

    Parameters
    ----------
    classifier : classifier
        A binary classifier to wrap.

    value : float, default 0.5
        Value of the `statistic` used to determine the threshold.

    statistic : {'precision', 'recall'}, default 'precision'
        Statistic to use to determine the threshold.

    pos_label : int or str, default 1
        The label of the positive class, to which `statistic` refers.

    test_size : float or int, default 0.1
        Size of the test data set, taken out of the data given to `fit`.
        - float: Proportion of the data, between 0.0 and 1.0.
        - int: Absolute number of samples.

    Attributes
    ----------
    threshold_ : float
        The threshold applied to `statistic` upon prediction.
    """
    STATS = {       # index, reduction func, unattainable value
        'precision': (0,     np.min,          1.1),
        'recall':    (1,     np.max,         -0.1),
    }

    def __init__(
            self,
            classifier: Classifier,
            value: float = 0.5,
            *,
            statistic: Literal['precision', 'recall'] = 'precision',
            pos_label: int | str = 1,
            test_size: float | int = 0.1,
    ):
        self.classifier = classifier
        self.value = value
        self.statistic = statistic
        self.pos_label = pos_label
        self.test_size = test_size

    def fit(self, X, y, **fit_params):
        if self.statistic not in self.STATS:
            raise ValueError(f"Unknown statistic `{self.statistic}`. Possible values: {', '.join(self.STATS)}.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, stratify=y)

        self.classifier.fit(X_train, y_train, **fit_params)
        self.probmask_ = self.classifier.classes_ == self.pos_label
        self.srt_ = np.argsort(self.probmask_)

        # these values can be used for post-adjustment of threshold
        self.X_test_probs_ = self._probs(X_test)
        self.y_test_values_ = y_test.values

        self._update_threshold()
        return self

    def set_params(self, **params):
        super().set_params(**params)
        if 'value' in params:
            try:
                check_is_fitted(self, ['X_test_probs_', 'y_test_values_'])
            except NotFittedError:
                pass
            else:
                self._update_threshold()
        return self

    def _update_threshold(self):
        prt = precision_recall_curve(self.y_test_values_, self.X_test_probs_, pos_label=self.pos_label)
        ind, reduce_func, unattainable = self.STATS[self.statistic]
        stat = prt[ind][:-1]
        thresholds = prt[-1]
        valid = np.append(thresholds[stat >= self.value], unattainable)
        self.threshold_ = reduce_func(valid)

    def predict(self, X):
        check_is_fitted(self, ['probmask_', 'srt_', 'threshold_'])
        return self.classifier.classes_[self.srt_][(self._probs(X) >= self.threshold_).astype(int)]

    @property
    def classes_(self):
        return self.classifier.classes_

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict_log_proba(self, X):
        if hasattr(self.classifier, 'predict_log_proba'):
            return self.classifier.predict_log_proba(X)
        return np.log(self.predict_proba(X))

    def _probs(self, X):
        return self.predict_proba(X)[:, self.probmask_]


class WeightedLinearOVRClassifier(BaseEstimator, LinearClassifierMixin, ClassifierMixin):
    """
    A classifier which wraps a linear classifier and performs a weighted OVR strategy between the classes.
    
    For each class, the classifier is fitted separately, in parallel.
    The weights are determined by a price matrix, giving the price for each classification mistake,
    and the class probability of occurrence.
    
    Usually, the price matrix is positive on the diagonal (reward points) and negative
    elsewhere (price for mistake).

    This classifier uses `metrics.confusion_score` as its score. It is the score appropriate for
    evaluating the classification results, taking into account both `price` and `classprobs`.

    Parameters
    ----------
    classifier : linear classifier
        A linear classifier to wrap.

    price : array-like or DataFrame, optional
        Pricing matrix, with shape (n_classes_, n_classes_).
        If an array is given, the classes are inferred from the training labels, in sorted order.
        If not provided, the pricing matrix has 1 on the diagonal and -1 elsewhere.

    classprobs : array-like or Series, optional
        Probability of appearance for each class.
        If an array is given instead of a Series, classes are assumed in the same manner
        as for `price`. If not provided, assigns equal probabilities to all classes.
        Given probabilities will be normalized internally.

    n_jobs : int, default 1
        Number of jobs to run in parallel.
    """
    def __init__(
            self,
            classifier: LinearClassifier,
            *,
            price: ArrayLike | pd.DataFrame | None = None,
            classprobs: ArrayLike | pd.Series | None = None,
            n_jobs: int = 1,
    ):
        self.classifier = classifier
        self.price = price
        self.classprobs = classprobs
        self.n_jobs = n_jobs

    def score(self, X, y, sample_weight=None):
        return confusion_score(y, self.predict(X), self.price, classprobs=self.classprobs, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self._predict_proba_lr(X)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def _check_price(self) -> pd.DataFrame:
        if isinstance(self.price, pd.DataFrame):
            return self.price.reindex(index=self.classes_, columns=self.classes_, copy=False, fill_value=0)
        if self.price is None:
            price = pd.DataFrame(np.full((len(self.classes_),) * 2, -1), index=self.classes_, columns=self.classes_)
            diag = np.arange(price.shape[0])
            price.values[diag, diag] = 1
            return price
        price = np.asarray(self.price)
        if price.ndim != 2 or price.shape[0] != price.shape[1] or price.shape[0] != len(self.classes_):
            raise ValueError(
                f"`price` must be square and have length {len(self.classes_)}; got shape {price.shape}."
            )
        return pd.DataFrame(price, index=self.classes_, columns=self.classes_)

    def _check_classprobs(self) -> pd.Series:
        if isinstance(self.classprobs, pd.Series):
            probs = self.classprobs.reindex(self.classes_, copy=False, fill_value=0)
        elif self.classprobs is None:
            probs = pd.Series(1, index=self.classes_)
        else:
            probs = np.asarray(self.classprobs)
            if probs.ndim != 1 or len(probs) != len(self.classes_):
                raise ValueError(
                    "Dimensions mismatch for `classprobs`: "
                    f"Should be 1-dim with length {len(self.classprobs)}; got {probs.shape}."
                )
            probs = pd.Series(probs, index=self.classes_)
        return probs / probs.sum()

    def fit(self, X, y):
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=np.float64, order="C")
        self.classes_ = np.unique(y)
        price = self._check_price()
        probs = self._check_classprobs()
        fit_single = delayed(_fit_single_linear_ovr)
        fitted = Parallel(self.n_jobs)(
            fit_single(clone(self.classifier), X, y, price[c] * probs) for c in self.classes_
        )
        self.coef_ = np.empty((len(self.classes_), X.shape[1]), dtype=np.float64)
        self.intercept_ = np.empty(len(self.classes_), dtype=np.float64)
        for i, clf in enumerate(fitted):
            scale = price[self.classes_[i]].abs().mean()
            norm = np.linalg.norm(clf.coef_[0])
            if norm != 0:
                scale /= norm
            self.coef_[i] = clf.coef_[0] * scale
            self.intercept_[i] = clf.intercept_[0] * scale
        return self


def _fit_single_linear_ovr(clf: LinearClassifier, X: NDArray, y: NDArray, classweights: pd.Series) -> LinearClassifier:
    yy = np.where(np.in1d(y, classweights.index[classweights > 0]), 1, -1)
    w = classweights.abs()[y].values
    return clf.fit(X, yy, sample_weight=w)
