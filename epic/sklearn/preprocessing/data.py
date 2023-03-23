import numpy as np
import pandas as pd

from scipy import optimize
from typing import overload
from numpy.typing import ArrayLike, NDArray
from collections.abc import Mapping, Hashable, Iterable

from sklearn.utils import tosequence
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin

from epic.common.general import is_iterable, coalesce

from ..utils import check_dataframe


class BinningTransformer(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """
    Discretize numerical values into bins.

    Columns to be binned can be specified, other columns are left as is.
    NaN values are preserved by the transformer.

    Parameters
    ----------
    bins : dictionary, optional
        A mapping from columns to the bins to use for that column.
        Columns not present will not be transformed.

        Keys : int or string
            Column indices.
            If the input to `transform` is a DataFrame, can also be column names.

        Values : iterable, int or string of the form "klog"
            The bins to use.
            - Iterable: Must be monotonic. An iterable of length n implies n+1 bins:
                [(-inf, bins[0]), [bins[0], bins[1]), [bins[1], bins[2]), ..., [bins[n], inf)]
            - Integer: The number of evenly-spaced bins between the minimum and maximum of the data.
            - "<k>log": Here, 'k' is an integer (e.g. "10log"). In this case, k logarithmically-spaced
                        bins between the minimum and maximum of the data will be used.

    copy : bool, default True
        Whether to copy the data upon transformation or perform the action in-place.
    """
    def __init__(self, bins: Mapping[Hashable, Iterable | int | str] | None = None, *, copy: bool = True):
        self.bins = bins
        self.copy = copy

    @staticmethod
    def _indexer(X, index):
        if isinstance(X, pd.DataFrame):
            if isinstance(index, int):
                return X.iloc
            return X.loc
        return X

    @classmethod
    def _get_col(cls, X, index):
        return pd.Series(cls._indexer(X, index)[:, index])

    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        if not isinstance(X, pd.DataFrame):
            X = check_array(X, force_all_finite=False)
        self.bins_ = {}
        for ind, bins in coalesce(self.bins, {}).items():
            if is_iterable(bins):
                self.bins_[ind] = tosequence(bins)
            else:
                log = False
                if isinstance(bins, str) and bins.endswith('log'):
                    try:
                        bins = int(bins[:-3])
                    except ValueError:
                        pass
                    else:
                        log = True
                if isinstance(bins, int) and bins > 1:
                    col = self._get_col(X, ind)
                    min_val = col.min()
                    max_val = col.max()
                    if log:
                        min_val = np.log10(min_val)
                        max_val = np.log10(max_val)
                        space = np.logspace
                    else:
                        space = np.linspace
                    self.bins_[ind] = space(min_val, max_val, num=bins + 1)[1:-1]
                else:
                    raise ValueError(f"Invalid format for `bins`: {bins}.")
        return self

    def transform(self, X):
        check_is_fitted(self, 'bins_')
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        if not isinstance(X, pd.DataFrame):
            X = check_array(X, force_all_finite=False)
        Xt = X.copy() if self.copy else X
        for ind, bins in self.bins_.items():
            col = self._get_col(X, ind)
            colt = np.digitize(col, bins)
            if (is_na := col.isnull()).any():
                colt = colt.astype(float)
                colt[is_na] = np.NaN
            self._indexer(Xt, ind)[:, ind] = colt
        return Xt

    def _more_tags(self):
        return {"allow_nan": True}


class YeoJohnsonTransformer(BaseEstimator, _OneToOneFeatureMixin):
    """
    Apply the Yeo-Johnson transform,[1]_ which changes the distribution of the data to be
    more like a normal distribution.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Power_transform
    """
    def fit_transform(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X, given_df = check_dataframe(X)
        self.lambdas_ = np.empty(X.shape[1], dtype=np.double)
        Xt = np.empty_like(X, dtype=np.double)
        for i, feature in enumerate(X.values.T):
            ft, lambda_ = self.yeo_johnson_transform(feature)
            Xt[:, i] = ft
            self.lambdas_[i] = lambda_
        return pd.DataFrame(Xt, index=X.index, columns=X.columns) if given_df else Xt

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'lambdas_')
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X, given_df = check_dataframe(X)
        Xt = self.yeo_johnson_transform(X.values, self.lambdas_)
        return pd.DataFrame(Xt, index=X.index, columns=X.columns) if given_df else Xt

    @overload
    @classmethod
    def yeo_johnson_transform(cls, data: ArrayLike, lambda_: None = None) -> tuple[NDArray[np.double], float]: ...

    @overload
    @classmethod
    def yeo_johnson_transform(cls, data: ArrayLike, lambda_: NDArray) -> NDArray[np.double]: ...

    @classmethod
    def yeo_johnson_transform(cls, data, lambda_=None):
        """
        Implementation of the Yeo-Johnson transform.[1]_
        This generalizes the Box-Cox transform, which works only for positive values, to all values.

        The function can either apply the transform given lambda, or find the best lambda
        in terms of maximum likelihood best fit to a normal distribution.

        Parameters
        ----------
        data : array-like
            Data to be transformed.

        lambda_ : array-like, optional
            Lambda parameter of the transform.
            - If provided, the transform is applied.
              The shapes of `lambda_` and `data` should be compatible (broadcastable).
              The transform is applied for each datum, for each lambda.
            - If not provided, the best-fit lambda is estimated.
              The transform is applied using this lambda, and both the transformed data and the
              best-fit lambda are returned.

        Returns
        -------
        transformed : numpy array
            Transformed data.
            Shape is a broadcast of `data` and `lambda_`, if given, or same as `data` otherwise.

        best_lambda : float
            Best-fit lambda.
            Only returned if `lambda_` was not provided.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Power_transform
        """
        data = np.asarray(data)
        if data.size == 0:
            return data
        if lambda_ is not None:
            return cls._yeo_johnson(lambda_, data)
        best_lambda = optimize.brent(cls._neg_yeo_johnson_loglikelihood, brack=(0, 2), args=(data,))
        if is_iterable(best_lambda):
            best_lambda = best_lambda[0]
        return cls._yeo_johnson(best_lambda, data), best_lambda

    @staticmethod
    def _yeo_johnson(lambda_: ArrayLike, data: ArrayLike) -> NDArray[np.double]:
        """
        Apply the transform.
        """
        lambda_, data = np.broadcast_arrays(lambda_, data)
        l0 = lambda_ == 0
        l2 = lambda_ == 2
        result = np.empty_like(lambda_, dtype=np.double)

        m = (~l0) & (data >= 0)
        xm, lm = data[m], lambda_[m]
        result[m] = ((xm + 1.) ** lm - 1.) / lm

        m = l0 & (data >= 0)
        result[m] = np.log1p(data[m])

        m = (~l2) & (data < 0)
        xm, lm = data[m], lambda_[m]
        result[m] = -((-xm + 1.) ** (2. - lm) - 1.) / (2. - lm)

        m = l2 & (data < 0)
        result[m] = -np.log1p(-data[m])

        return result

    @classmethod
    def _neg_yeo_johnson_loglikelihood(cls, lambda_: ArrayLike, data: ArrayLike) -> NDArray[np.double]:
        """
        Negative log-likelihood of the transform.
        """
        shape = np.asarray(lambda_).shape
        lambda_, data = np.broadcast_arrays(lambda_, data)
        shape = (1,) * (lambda_.ndim - len(shape)) + shape
        for ax in range(lambda_.ndim - 1, -1, -1):
            if shape[ax] == 1 and lambda_.shape[ax] != 1:
                break
        else:
            raise ValueError("Shape mismatch for `lambda` and `data`.")
        yj = cls._yeo_johnson(lambda_, data)
        miu = yj.mean(axis=ax).reshape(shape)
        var = ((yj - miu) ** 2).mean(axis=ax).reshape(shape)
        m = data >= 0
        log = np.empty_like(lambda_, dtype=np.double)
        log[m] = np.log1p(data[m])
        log[~m] = -np.log1p(-data[~m])
        return (np.log(2 * np.pi * var) + 1.) * (yj.shape[ax] / 2.) - ((lambda_ - 1) * log).sum(axis=ax).reshape(shape)


# A stand-alone version of the transform
yeo_johnson_transform = YeoJohnsonTransformer.yeo_johnson_transform


class TailChopper(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """
    Replace outlier values, i.e. values that are too far from the mean, with
    boundary values.

    Parameters
    ----------
    sigma : float, default 5
        Number of standard deviations away from the mean, in each direction, at which to set the bound.
        Cannot be negative.
    """
    def __init__(self, sigma: float = 5):
        self.sigma = sigma

    def fit(self, X, y=None):
        if self.sigma < 0:
            raise ValueError(f"`sigma` must be non-negative; got {self.sigma:g}.")
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X, _ = check_dataframe(X)
        mean = X.mean()
        std = X.std()
        self.bounds_ = pd.DataFrame({
            'upper': mean + self.sigma * std,
            'lower': mean - self.sigma * std,
        })
        return self

    def transform(self, X):
        check_is_fitted(self, 'bounds_')
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X, given_df = check_dataframe(X)
        Xt = X.clip(lower=self.bounds_.lower, upper=self.bounds_.upper, axis=1)
        return Xt if given_df else Xt.values

    def _more_tags(self):
        return {"allow_nan": True}
