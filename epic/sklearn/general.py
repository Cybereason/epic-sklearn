import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import TypeVar, Protocol
from collections.abc import Collection
from abc import ABCMeta, abstractmethod

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import check_array_permissive

T = TypeVar('T')
BC = TypeVar('BC', bound='BaseClassifier')
RC = TypeVar('RC', bound='RandomClassifier')


class BaseClassifier(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):
    """
    An alternative to using `BaseEstimator` for classifiers.

    Any subclass must define two methods:
    - _predict_proba: Predicts raw probabilities from features.
    - classes_: A property which returns an array of classes after training.
                The order of the classes should match the second dimension of
                the probabilities returned by `_predict_proba`.

    From these two methods everything else is calculated automatically.
    """
    def fit(self: BC, X, y=None) -> BC:
        return self

    def _predict(self, X) -> NDArray:
        return self.classes_[self._predict_proba(X).argmax(axis=1)]

    def predict(self, X) -> NDArray | pd.Series:
        pred = self._predict(X)
        if isinstance(X, pd.DataFrame):
            return pd.Series(pred, index=X.index)
        return pred

    def fit_predict(self, X, y=None) -> NDArray | pd.Series:
        return self.fit(X, y).predict(X)

    @abstractmethod
    def _predict_proba(self, X) -> NDArray:
        pass

    def predict_proba(self, X) -> NDArray | pd.DataFrame:
        probs = self._predict_proba(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(probs, index=X.index, columns=self.classes_)
        return probs

    def predict_log_proba(self, X) -> NDArray | pd.DataFrame:
        return np.log(self.predict_proba(X))

    @property
    @abstractmethod
    def classes_(self) -> NDArray:
        pass


class ConstantClassifier(BaseClassifier):
    """
    A classifier which always predicts a constant.

    Parameters
    ----------
    constant : object, default 0
        Constant to predict.
    """
    def __init__(self, constant=0):
        self.constant = constant

    def _predict_proba(self, X):
        return np.ones((check_array_permissive(X).shape[0], 1), dtype=float)

    @property
    def classes_(self):
        return np.array([self.constant])

    def _more_tags(self):
        return {"stateless": True}


class RandomClassifier(BaseClassifier):
    """
    A classifier which predicts one of its classes at random.

    Parameters
    ----------
    classes : collection
        The possible classes for the estimator.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator.
    """
    def __init__(self, classes: Collection = (0, 1), random_state: int | np.random.RandomState | None = None):
        self.classes = classes
        self.random_state = random_state

    def fit(self: RC, X, y=None) -> RC:
        self.random_ = check_random_state(self.random_state)
        return self

    def _predict_proba(self, X):
        probs = self.random_.random((check_array_permissive(X).shape[0], len(self.classes) - 1))
        return np.hstack((probs, 1 - probs.sum(axis=1).reshape((-1, 1))))

    @property
    def classes_(self):
        return np.asarray(self.classes)

    def _more_tags(self):
        return {"non_deterministic": True}


class Estimator(Protocol):
    """
    A protocol for an Estimator.
    For this purpose, whatever defines a `fit` method is considered an estimator.
    """
    def fit(self: T, X, y=None, **fit_params) -> T: ...


class Classifier(Estimator):
    """A protocol for a Classifier."""
    classes_: NDArray

    def predict_proba(self, X): ...
    def predict(self, X): ...


class LinearClassifier(Classifier):
    """A protocol for a Linear Classifier."""
    coef_: NDArray
    intercept_: NDArray


class FitPredictMixin:
    """Mixin class to add a naive implementation of `fit_predict`."""
    def fit_predict(self: Classifier, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).predict(X)
