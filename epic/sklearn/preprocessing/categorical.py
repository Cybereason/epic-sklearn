import warnings
import numpy as np
import pandas as pd

from typing import Literal
from cytoolz import compose
from functools import partial
from scipy import sparse as sp
from operator import itemgetter
from collections import Counter
from numpy.typing import DTypeLike
from collections.abc import Iterable

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin

from epic.common.general import is_iterable, to_list, to_iterable

from ..utils import check_dataframe
from .label import MultiLabelEncoder


class FrequencyTransformer(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """
    A transformer which replaces each value with its occurrence frequency within the feature.
    Values encountered in the transform stage but not in the fit stage are assumed to appear only once.

    Parameters
    ----------
    impute : bool, default True
        Whether to transform NaNs to the median frequency.
        If False, NaNs will be left as is.
    """
    def __init__(self, *, impute: bool = True):
        self.impute = impute

    @staticmethod
    def _histogram(series: pd.Series) -> tuple[pd.Series, float]:
        return series.value_counts(normalize=True), 1 / len(series)

    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X, _ = check_dataframe(X)
        self.map_ = {}
        for name, feature in X.items():
            hist, single_freq = self._histogram(feature)
            hist[np.NaN] = hist[hist.index[hist.cumsum() >= 0.5][0]] if len(hist) > 1 else 0.5
            self.map_[name] = (hist, single_freq)
        return self

    def _transform_feature(self, feature: pd.Series) -> pd.Series:
        hist, new_val = self.map_[feature.name]
        ft = feature.map(hist).fillna(new_val)
        if not self.impute:
            ft[feature.isnull()] = np.NaN
        return ft

    def transform(self, X):
        check_is_fitted(self, 'map_')
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        Xt = check_dataframe(X)[0].apply(self._transform_feature).reindex(columns=list(self.map_))
        if isinstance(X, pd.DataFrame):
            return Xt
        if isinstance(X, pd.Series):
            return Xt.iloc[:, 0]
        return Xt.values

    def _more_tags(self):
        return {"allow_nan": True}


class FrequencyListTransformer(FrequencyTransformer):
    """
    A transformer which is similar to `FrequencyTransformer` but meant for features which are lists of values.

    Each value within the list is replaced with its overall occurrence frequency in the data.
    Values encountered in the transform stage but not in the fit stage are assumed to appear only once.

    Parameters
    ----------
    impute : bool, default True
        Whether to impute NaN values appearing *inside* the list for each sample.
        If True, NaNs will be replaced with the median frequency.
        Regardless, samples which are NaN (and not a list of values) will be left as is.

    See Also
    --------
    FrequencyTransformer : Transform a single value per feature per sample.
    """
    @staticmethod
    def _histogram(series: pd.Series) -> tuple[pd.Series, float]:
        cnt = Counter()
        series.map(cnt.update, na_action='ignore')
        cnt.pop(np.NaN, None)
        hist = pd.Series(cnt).sort_values(ascending=False)
        n_values = hist.sum()
        return hist / n_values, 1 / n_values

    def _transform_values(self, values, hist, new_val):
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        vt = values.map(hist).fillna(new_val)
        if not self.impute:
            vt[values.isnull()] = np.NaN
        return vt.tolist()

    def _transform_feature(self, feature: pd.Series) -> pd.Series:
        hist, new_val = self.map_[feature.name]
        return feature.map(partial(self._transform_values, hist=hist, new_val=new_val), na_action='ignore')


class ListStatisticsTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate statistics on lists of values.

    Each sample of each feature is expected to be a list of values.
    Upon transformation, on each such list, four statistics are computed: mean, std, max and min.

    Parameters
    ----------
    keep_series_name : bool, default False
        Only relevant if transforming a single Series.
        If False, the series name is not kept and the resulting DataFrame columns are
        the statistics calculated.
    """
    FUNCS = 'mean', 'std', 'max', 'min'

    def __init__(self, *, keep_series_name: bool = False):
        self.keep_series_name = keep_series_name

    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X):
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        Xdf, _ = check_dataframe(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Xt = pd.concat({feature_name: feature.apply(
                lambda x: pd.Series({f: getattr(np, f'nan{f}')(np.r_[x, np.nan]) for f in self.FUNCS}).fillna(0)
            ) for feature_name, feature in Xdf.items()}, axis=1)
        if isinstance(X, pd.Series) and not self.keep_series_name:
            Xt.columns = Xt.columns.droplevel(0)
            return Xt
        if isinstance(X, pd.Series | pd.DataFrame):
            Xt.columns = Xt.columns.map('_'.join)
            return Xt
        return Xt.values

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, 'feature_names_in_'):
            return
        return np.array([f'{f}_{x}' for f in self.feature_names_in_ for x in self.FUNCS])

    def _more_tags(self):
        return {"allow_nan": True}


class CategoryEncoder(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """
    A transformer which encodes categorical features into integers.

    NaNs are left as is, and values encountered in `transform` but not in `fit` are
    transformed into NaN.
    """
    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X, _ = check_dataframe(X)
        self.encoders_ = {n: LabelEncoder().fit(f.dropna()) for n, f in X.items()}
        return self

    def _transform_feature(self, feature: pd.Series) -> pd.Series:
        encoder = self.encoders_[feature.name]
        known = feature.isin(encoder.classes_)  # also not NaN
        if known.all():
            data = encoder.transform(feature)
        else:
            data = np.full(len(feature), fill_value=np.NaN)
            data[known] = encoder.transform(feature[known])
        return pd.Series(data, index=feature.index, name=feature.name)

    def transform(self, X):
        check_is_fitted(self, 'encoders_')
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X, is_df = check_dataframe(X)
        Xt = X.apply(self._transform_feature)
        return Xt if is_df else Xt.values

    def _more_tags(self):
        return {"allow_nan": True}


class ManyHotEncoder(BaseEstimator):
    """
    A transformer which is similar to `sklearn.preprocessing.OneHotEncoder` but meant for
    features which are lists of values.

    Each list is transformed into a binary vector with a "1" for each of the values.
    The transformer passes along non-categorical features unchanged, but has them concatenated
    to the right of all transformed categorical features. The resulting matrix can be sparse
    or dense, DataFrame or bare.

    Parameters
    ----------
    categorical_features : 'all', iterable of ints or iterable of strings, default 'all'
        Which of the features are categorical.
        - 'all': All of them.
        - Iterable of ints: Column indices.
        - Iterable of strings: Column names.

    dtype : dtype-like, default float
        Data type of the resulting transformed matrix.

    sparse : bool, default True
        Whether to transform into a sparse matrix or DataFrame.

    handle_unknown : {'error', 'ignore'}, default 'error'
        Whether to raise an exception upon transforming values unseen during the fitting
        stage or simply ignore such values.

    df_out : {False, 'single', 'multi'}, default False
        Whether to transform into a DataFrame, and the kind of columns index it should have.
        - False: Output is a numpy array or scipy sparse matrix, depending on `sparse`.
        - 'single': Output is a DataFrame with index in the format 'ftr__value', where
                    `ftr` is the original feature name and `value` is the encoded value.
        - 'multi': Output is a DataFrame with a two-level MultiIndex for feature names
                   and values.
        In both DataFrame options, if `sparse` is True, a sparse DataFrame is returned.

    allow_singles : bool, default False
        If True, each sample can also be a single value, which will be converted
        to a list of length 1.

    See Also
    --------
    sklearn.preprocessing.OneHotEncoder : Encode a single value as a binary vector with a single "1".
    """
    def __init__(
            self,
            categorical_features: Literal['all'] | Iterable[int] | Iterable[str] = "all",
            *,
            dtype: DTypeLike = np.float64,
            sparse: bool = True,
            handle_unknown: Literal['error', 'ignore'] = 'error',
            df_out: Literal[False, 'single', 'multi'] = False,
            allow_singles: bool = False,
    ):
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.df_out = df_out
        self.allow_singles = allow_singles

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X, _ = check_dataframe(X)
        if self.categorical_features == 'all':
            self.cat_ = X.columns.tolist()
        elif not is_iterable(self.categorical_features):
            raise ValueError(
                "`categorical_features` must either be an iterable, or 'all'; "
                f"was given {self.categorical_features}."
            )
        elif all(isinstance(x, int) for x in self.categorical_features):
            self.cat_ = X.columns[self.categorical_features].to_list()
        elif len(self.categorical_features) != len(set(self.categorical_features)):
            raise ValueError("`categorical_features` must be unique.")
        elif any(x not in X for x in self.categorical_features):
            raise ValueError("All categorical features must appear in `X`.")
        else:
            self.cat_ = to_list(self.categorical_features)
        self.noncat_ = X.columns.difference(self.cat_).to_list()
        return self._hstack(self._fit_transform(X[self.cat_]), X[self.noncat_])

    def _hstack(self, Xcat: np.ndarray | sp.spmatrix, Xnotcat: pd.DataFrame):
        if self.df_out not in {False, 'single', 'multi'}:
            raise ValueError(f"Invalid `df_out`: '{self.df_out}'.")
        if self.df_out:
            index = Xnotcat.index
            columns = self.get_feature_names(tuples=False) if self.df_out == 'single' \
                else pd.MultiIndex.from_tuples(self.get_feature_names(tuples=True), names=('features', 'values'))
        if sp.issparse(Xcat):
            X = sp.hstack((Xcat, Xnotcat.values), format='csr', dtype=self.dtype)
            return pd.DataFrame.sparse.from_spmatrix(X, index=index, columns=columns) if self.df_out else X
        if self.df_out:
            Xcatdf = pd.DataFrame(Xcat, index=index)
            Xdf = pd.concat((Xcatdf, Xnotcat), axis=1)
            Xdf.columns = columns
            return Xdf
        return np.hstack((Xcat, Xnotcat.values.astype(self.dtype, copy=False)))

    def transform(self, X):
        self._check_is_fitted()
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X, _ = check_dataframe(X)
        return self._hstack(self._transform(X[self.cat_]), X[self.noncat_])

    def _fit_transform(self, X: pd.DataFrame) -> np.ndarray | sp.spmatrix:
        self.encoders_ = []
        feature_indices = [0]
        rows = []
        cols = []

        for _, feature in X.items():
            enc = MultiLabelEncoder(allow_singles=self.allow_singles)
            self.encoders_.append(enc)
            ft: pd.Series = enc.fit_transform(feature)

            n_classes = len(enc.classes_)
            f_ind = feature_indices[-1]
            feature_indices.append(f_ind + n_classes)

            r, c = self._calc_rows_cols(ft, f_ind)
            rows.extend(r)
            cols.extend(c)

        self.feature_indices_ = np.array(feature_indices, dtype=int)
        Xt = self._build_sparse_mat(X.shape[0], rows, cols)
        return Xt if self.sparse else Xt.toarray()

    @staticmethod
    def _calc_rows_cols(col_lists: pd.Series, feature_index: int):
        col_lists = col_lists.reset_index(drop=True).dropna()
        if col_lists.empty:
            return (), ()
        lengths = col_lists.map(len)
        return lengths.index.repeat(lengths), np.concatenate(col_lists.values) + feature_index

    def _build_sparse_mat(self, n_samples, rows, cols):
        data = np.ones_like(rows, dtype=self.dtype)
        return sp.coo_matrix((data, (rows, cols)), shape=(n_samples, self.feature_indices_[-1]))

    def _check_is_fitted(self):
        check_is_fitted(self, ['cat_', 'noncat_', 'encoders_', 'feature_indices_'])

    def _transform(self, X: pd.DataFrame):
        if self.handle_unknown not in ('error', 'ignore'):
            raise ValueError(f"`handle_unknown` should be either 'error' or 'unknown'; got {self.handle_unknown}.")
        rows = []
        cols = []
        for enc, name, f_ind in zip(self.encoders_, X, self.feature_indices_[:-1]):
            feature: pd.Series = X[name]
            if self.handle_unknown == 'error':
                try:
                    ft = enc.transform(feature)
                except ValueError as exc:
                    raise ValueError(f"Error when transforming feature {name}.") from exc
            else:
                classes = set(enc.classes_)
                filtered = feature.map(
                    compose(classes.intersection, to_iterable) if self.allow_singles else classes.intersection,
                    na_action='ignore',
                )
                ft = enc.transform(filtered)
            r, c = self._calc_rows_cols(ft, f_ind)
            rows.extend(r)
            cols.extend(c)
        Xt = self._build_sparse_mat(X.shape[0], rows, cols)
        return Xt if self.sparse else Xt.toarray()

    def _categorical_feature_columns(self, tuples):
        return [
            (ftr, val) if tuples else f'{ftr}__{val}'
            for ftr, enc in zip(self.cat_, self.encoders_)
            for val, index in sorted(enc.classes_.items(), key=itemgetter(1))
        ]

    def _noncategorical_feature_columns(self, tuples):
        return [(x, '') for x in self.noncat_] if tuples else self.noncat_

    def get_feature_names(self, tuples=False):
        self._check_is_fitted()
        return self._categorical_feature_columns(tuples) + self._noncategorical_feature_columns(tuples)

    def get_feature_names_out(self, input_features=None):
        return self.get_feature_names()
