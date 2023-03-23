import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from collections import defaultdict
from joblib import Parallel, delayed
from collections.abc import Callable, Iterable

from sklearn.utils import tosequence
from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition

from .general import Estimator


class SampleSplitter(_BaseComposition, TransformerMixin):
    """
    Concatenates results vertically for multiple transformer or estimator objects.

    This is a concept comparable to FeatureUnion.
    The major difference is that samples, and not features, are split into distinct groups.

    Various estimator actions (e.g. predict or predict_proba) are passed on to the estimators.
    For every input `X`, only estimators of non-empty splits are called.

    Note: If a lambda is used as the split function, then the resulting transformer will
    not be pickleable.

    Parameters
    ----------
    split_func : callable
        The callable to use for assigning the transformation path for each vector.
        It should take as input a single sample (1D-array or Series, depending on the input to the estimator)
        and return an estimator name.

    estimators: iterable of (string, estimator) tuples
        Estimator objects to be applied to the data according to split.
        The first item of each tuple is the name of the estimator.

    n_jobs: int, default 1
        Number of jobs to run in parallel.

    **split_func_kwargs:
        Forwarded to `split_func`.
    """
    def __init__(
            self,
            split_func: Callable[[ArrayLike | pd.Series], str],
            estimators: Iterable[tuple[str, Estimator]],
            n_jobs: int = 1,
            **split_func_kwargs,
    ):
        super().__init__()
        self.split_func = split_func
        self.estimator_list = tosequence(estimators)
        self.n_jobs = n_jobs
        self.split_func_kwargs = split_func_kwargs
        self._validate_estimators()
        self.split_results = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('estimator_list', deep=deep)

    def set_params(self, **kwargs):
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('estimator_list', **kwargs)

    def _validate_estimators(self):
        names, estimators = zip(*self.estimator_list)
        self._validate_names(names)
        self._validate_estimators_have_method('fit', accept_none=True)

    def _validate_estimators_have_method(self, method_name, *, accept_none):
        for name, est in self.estimator_list:
            if est is None:
                if not accept_none:
                    raise AttributeError(f"Cannot call method `{method_name}`; estimator '{name}' is None.")
            elif not hasattr(est, method_name):
                raise AttributeError(f"Cannot call method `{method_name}`; estimator '{name}' does not implement it.")

    @property
    def _estimator_type(self):
        for name, est in self.estimator_list:
            if hasattr(est, '_estimator_type'):
                return est._estimator_type

    @property
    def estimators(self) -> dict[str, Estimator]:
        return dict(self.estimator_list)

    def _split(self, X):
        if isinstance(X, pd.DataFrame):
            split = X.apply(self.split_func, axis=1, **self.split_func_kwargs)
        else:
            split = pd.Series(np.apply_along_axis(self.split_func, 1, X, **self.split_func_kwargs))

        if split.isnull().any():
            raise ValueError("Result of `split_func` must not include null.")

        if unknown_keys := set(split.unique()) - set(self.estimators):
            raise ValueError(f"Unknown estimator names in `split_func` result: {unknown_keys}")

        self.split_results = split
        return {name: index.values for name, index in pd.RangeIndex(len(split)).groupby(split).items()}

    def _merge(self, results, groups, index=None, fill=None):
        pdresults = []
        for res, (name, ind) in zip(results, groups.items()):
            if isinstance(res, (pd.DataFrame, pd.Series)) and index is not None:
                ilocs = index.get_indexer(res.index)
                if set(ilocs) != set(ind):
                    raise ValueError(f"Estimator '{name}' returned result with unexpected index.")
                res.index = ilocs
            else:
                res = np.asanyarray(res)
                if res.ndim == 1 or res.ndim == 2 and res.shape[1] == 1:
                    res = pd.Series(data=res.flatten(), index=ind)
                elif res.ndim == 2:
                    res = pd.DataFrame(data=res, index=ind,
                                       columns=None if fill is None else self.estimators[name].classes_)
                else:
                    raise ValueError(f"Result has unexpected number of dimensions: {res.ndim}.")
            if res.shape[0] != len(ind):
                raise ValueError(f"Numer of results is {res.shape[0]}, expected {len(ind)}.")
            pdresults.append(res)

        if len({x.ndim for x in pdresults}) > 1:
            raise ValueError("Mismatch in number of dimensions for results of different groups.")

        # Index is now row numbers in the input dataset
        merged = pd.concat(pdresults).sort_index()
        if fill is not None:
            merged = merged.reindex(columns=self.classes_)
            merged.fillna(fill, inplace=True)
        if index is None:
            return merged.values
        merged.index = index
        return merged

    @staticmethod
    def _idx(obj, ind):
        return None if obj is None else getattr(obj, 'iloc', obj)[ind]

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model.

        Fit all the estimators, splitting the samples between them according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.
            Training data.

        y : iterable, optional
            Training targets.

        **fit_params :
            Parameters passed to the `fit` method of each estimator, where each parameter name
            is prefixed such that parameter 'p' for estimator 'e' has key 'e__p'.

        Returns
        -------
        self : SampleSplitter
            This estimator.
        """
        self._validate_estimators()
        estimators = self.estimators
        groups = self._split(X)
        params = self._split_params(fit_params)
        fit = delayed(_fit_one)

        ests = Parallel(n_jobs=self.n_jobs)(
            fit(estimators[name], self._idx(X, ind), self._idx(y, ind), **params[name])
            for name, ind in groups.items() if estimators[name] is not None
        )
        self._update_estimator_list(ests, [n for n in groups if estimators[n] is not None])
        return self

    @staticmethod
    def _split_params(unified_params):
        params = defaultdict(dict)
        for pname, pval in unified_params.items():
            estname, param = pname.split('__', 1)
            params[estname][param] = pval
        return params

    def _update_estimator_list(self, estimators, names):
        if len(estimators) != len(names):
            raise ValueError(f"Length mismatch: n_estimators = {len(estimators)}; n_names = {len(names)}.")
        estdict = self.estimators
        estdict.update(zip(names, estimators))
        self.estimator_list[:] = list(estdict.items())  # The order doesn't matter

    def _infer(self, method_name, X, accept_none=False, fill=None):
        self._validate_estimators_have_method(method_name, accept_none=accept_none)
        groups = self._split(X)
        estimators = self.estimators
        infer = delayed(_infer_one)
        Xs = Parallel(n_jobs=self.n_jobs)(
            infer(estimators[n], method_name, self._idx(X, ind)) for n, ind in groups.items()
        )
        return self._merge(Xs, groups, index=X.index if isinstance(X, pd.DataFrame) else None, fill=fill)

    def _fit_infer(self, method_name, X, y=None, accept_none=False, fill=None, **fit_params):
        self._validate_estimators()
        self._validate_estimators_have_method(method_name, accept_none=accept_none)
        estimators = self.estimators
        groups = self._split(X)
        params = self._split_params(fit_params)
        fit_infer = delayed(_fit_infer_one)

        results = Parallel(n_jobs=self.n_jobs)(
            fit_infer(estimators[name], method_name, self._idx(X, ind), self._idx(y, ind), **params[name])
            for name, ind in groups.items()
        )
        Xs, ests = zip(*results)
        self._update_estimator_list(ests, list(groups.keys()))
        return self._merge(Xs, groups, index=X.index if isinstance(X, pd.DataFrame) else None, fill=fill)

    def transform(self, X):
        """
        Apply transforms of all estimators, splitting the samples between them according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.
            Data to transform.

        Returns
        -------
        array-like, shape (n_samples, n_transformed_features)
        """
        return self._infer('transform', X, accept_none=True)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the model and apply transforms of all estimators, splitting the samples between them
        according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.
            Training data.

        y : iterable, optional
            Training targets.

        **fit_params :
            Parameters passed to the `fit` method of each estimator, where each parameter name
            is prefixed such that parameter 'p' for estimator 'e' has key 'e__p'.

        Returns
        -------
        array-like, shape (n_samples, n_transformed_features)
        """
        return self._fit_infer('transform', X, y=y, accept_none=True, **fit_params)

    def predict(self, X):
        """
        Predict using all estimators, splitting the samples between them according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
        """
        return self._infer('predict', X)

    def fit_predict(self, X, y=None, **fit_params):
        """
        Fit the model and predict using all estimators, splitting the samples between them
        according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.
            Training data.

        y : iterable, default=None
            Training targets.

        **fit_params :
            Parameters passed to the `fit` method of each estimator, where each parameter name
            is prefixed such that parameter 'p' for estimator 'e' has key 'e__p'.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
        """
        return self._fit_infer('predict', X, y=y, accept_none=False, **fit_params)

    def predict_proba(self, X):
        """
        Predict probabilities using all estimators, splitting the samples between them according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
        """
        return self._infer('predict_proba', X, fill=0)

    def decision_function(self, X):
        """
        Calculate decision function on all estimators, splitting the samples between them
        according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
        """
        return self._infer('decision_function', X)

    def predict_log_proba(self, X):
        """
        Predict log-probabilities using all estimators, splitting the samples between them
        according to the split_func.

        Parameters
        ----------
        X : array-like or DataFrame.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
        """
        return self._infer('predict_log_proba', X, fill=np.NINF)

    @property
    def classes_(self):
        return np.unique(np.concatenate([getattr(est, 'classes_', []) for _, est in self.estimator_list]))


def _fit_one(estimator, X, y, **fit_params):
    return estimator.fit(X, y, **fit_params)


def _fit_infer_one(estimator, method_name, X, y, **fit_params):
    fit_infer = 'fit_' + method_name
    if estimator is None:
        res = X
    elif hasattr(estimator, fit_infer):
        res = getattr(estimator, fit_infer)(X, y, **fit_params)
    else:
        res = getattr(estimator.fit(X, y, **fit_params), method_name)(X)
    return res, estimator


def _infer_one(estimator, method_name, X):
    return getattr(estimator, method_name)(X) if estimator is not None else X
