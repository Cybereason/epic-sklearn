import numpy as np
import pandas as pd

from logging import Handler
from functools import partial
from scipy import sparse as sp
from pandas._typing import Scalar
from typing import Literal, Generic, Any
from collections.abc import Hashable, Iterable, Callable, Mapping

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin

from ultima import Workforce
from ultima.backend import BackendArgument
from epic.logging import get_logger, ClassLogger
from epic.common.general import is_iterable, to_list
from epic.pandas.utils import pddump
from epic.pandas.sparse import SparseDataFrame
from epic.pandas.create import df_from_iterable, KT, DT, VT

from ..general import Estimator
from ..utils import check_dataframe


class SimpleTransformer(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
    """
    A base class for simple transformers.

    Such a transformer:
    - Doesn't change the number of features.
    - Doesn't change the names of the features.
    - Is stateless.
    - Allows NaNs in its input.
    Basically, it does something with the data, but otherwise passes it along as is.

    The SimpleTransformer itself does nothing.
    """
    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X):
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        return X

    def _more_tags(self):
        return {"stateless": True, "allow_nan": True}


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    """
    An estimator which wraps another estimator, and makes it suitable
    for DataFrame input and output.

    It makes sure that the columns present during the fitting stage are
    also there (and in the correct order) during predict or transform.
    It wraps the output as a DataFrame, keeping the index, and can also
    wrap it as a SparseDataFrame.

    Parameters
    ----------
    estimator : Estimator
        Estimator to wrap.
        It can be completely unaware to the existence of pandas.

    keep_colnames : bool, default True
        Whether `estimator` changes the columns of the data.
        If True, the output feature names are the column names of the input to `fit`.

    iterable_input : bool, default False
        If True, `estimator` expects an iterable of samples as input (e.g. CountVectorizer).
        The DataFrameWrapper then expects the input to be an iterable yielding
        (key, sample) pairs. The keys are collected into an Index, and the samples passed on
        to `estimator`.

    sparse_output_thin_wrap : bool, default False
        Only relevant if the output of applying `estimator` is a scipy sparse matrix.
        If True, the output is wrapped as a SparseDataFrame.
        If False, a DataFrame is built from it using `DataFrame.sparse.from_spmatrix`.
        See also `epic.pandas.sparse.SparseDataFrame`.
    """
    def __init__(
            self,
            estimator: Estimator,
            *,
            keep_colnames: bool = True,
            iterable_input: bool = False,
            sparse_output_thin_wrap: bool = False,
    ):
        self.estimator = estimator
        self.keep_colnames = keep_colnames
        self.iterable_input = iterable_input
        self.sparse_output_thin_wrap = sparse_output_thin_wrap

    @staticmethod
    def _conform_target(y: pd.Series, index: pd.Index) -> pd.Series:
        if not index.difference(y.index, sort=False).empty:
            raise ValueError("`X` contains samples not present in `y`.")
        if not y.index.difference(index, sort=False).empty:
            raise ValueError("`y` contains samples not present in `X`.")
        return y.loc[index]

    def _set_attrs(self, X: pd.DataFrame | None):
        # TODO: replace this with proper use of _check_feature_names
        self.input_columns_ = None if X is None else X.columns.copy()
        if self.keep_colnames and self.input_columns_ is not None:
            self.features_ = np.asarray(self.input_columns_)
        elif hasattr(self.estimator, "get_feature_names_out"):
            self.features_ = self.estimator.get_feature_names_out()
        elif hasattr(self.estimator, "get_feature_names"):
            self.features_ = self.estimator.get_feature_names()
        elif isinstance(self.estimator, Pipeline) and hasattr(self.estimator._final_estimator, "get_feature_names"):
            self.features_ = self.estimator._final_estimator.get_feature_names()
        else:
            self.features_ = None

    def _save_index(self, X):
        self._index = []
        for key, item in X:
            self._index.append(key)
            yield item

    def fit(self, X, y=None, **fit_params):
        if self.iterable_input:
            X = (v for k, v in X)
            Xdf = None
        else:
            Xdf = check_dataframe(X)[0]
            if isinstance(y, pd.Series):
                y = self._conform_target(y, Xdf.index)
        self.estimator.fit(X, y, **fit_params)
        self._set_attrs(Xdf)
        return self

    def _infer(self, method_name, X, kwargs):
        check_is_fitted(self, ('features_', 'input_columns_'))
        method = getattr(self.estimator, method_name)
        if self.iterable_input:
            X = self._save_index(X)
        else:
            X = check_dataframe(X)[0]
            if self.input_columns_ is not None:
                if not set(X.columns) >= set(self.input_columns_):
                    raise ValueError("There are input columns missing in the given data.")
                X = X.reindex(columns=self.input_columns_, copy=False)
            self._index = X.index
            if hasattr(X, 'sparse'):
                X = X.sparse.to_coo()
        return self._build_result(method(X, **kwargs))

    def _fit_infer(self, method_name, X, y, fit_params):
        method = getattr(self.estimator, f'fit_{method_name}')
        if self.iterable_input:
            X = self._save_index(X)
            Xdf = None
        else:
            Xdf = check_dataframe(X)[0]
            self._index = Xdf.index
            if isinstance(y, pd.Series):
                y = self._conform_target(y, Xdf.index)
        Xt = method(X, y, **fit_params)
        self._set_attrs(Xdf)
        return self._build_result(Xt)

    def _build_result(self, Xt):
        try:
            if Xt.shape[0] != len(self._index):
                raise ValueError(f"Expected result of length {len(self._index)}; got {Xt.shape[0]} instead.")
            if Xt.ndim > 2:
                raise ValueError(f"Unexpected number of dimensions for result: {Xt.ndim}.")
            if Xt.ndim == 2 and Xt.shape[1] == 1:
                Xt = Xt.squeeze(axis=1)
            if Xt.ndim == 1:
                if isinstance(Xt, pd.Series):
                    return Xt.reindex(index=self._index, copy=False)
                return pd.Series(Xt, index=self._index)
            for possible_columns in (self.features_, getattr(self.estimator, 'classes_', None)):
                if possible_columns is not None and Xt.shape[1] == len(possible_columns):
                    columns = possible_columns
                    break
            else:
                if self.features_ is None:
                    columns = None
                else:
                    raise ValueError(f"Unexpected number of columns for result: {Xt.shape[1]}.")
            if sp.issparse(Xt):
                builder = SparseDataFrame if self.sparse_output_thin_wrap else pd.DataFrame.sparse.from_spmatrix
                return builder(Xt, index=self._index, columns=columns)
            elif isinstance(Xt, pd.DataFrame):
                return Xt.reindex(index=self._index, columns=columns, copy=False)
            return pd.DataFrame(Xt, index=self._index, columns=columns)
        finally:
            del self._index

    def transform(self, X, **kwargs):
        return self._infer('transform', X, kwargs)

    def fit_transform(self, X, y=None, **fit_params):
        return self._fit_infer('transform', X, y, fit_params)

    def predict(self, X, **kwargs):
        return self._infer('predict', X, kwargs)

    def fit_predict(self, X, y=None, **fit_params):
        return self._fit_infer('predict', X, y, fit_params)

    def predict_proba(self, X, **kwargs):
        return self._infer('predict_proba', X, kwargs)

    def predict_log_proba(self, X, **kwargs):
        return self._infer('predict_log_proba', X, kwargs)

    def decision_function(self, X, **kwargs):
        return self._infer('decision_function', X, kwargs)

    def get_feature_names(self):
        check_is_fitted(self, 'features_')
        return self.features_

    def get_feature_names_out(self, input_features=None):
        if self.keep_colnames and input_features is not None:
            return np.asarray(input_features)
        return self.get_feature_names()

    def _wrap_attribute(self, name):
        if hasattr(self.estimator, name):
            return getattr(self.estimator, name)
        raise AttributeError(f"Object has no attribute '{name}'.")

    @property
    def _estimator_type(self):
        return self._wrap_attribute('_estimator_type')

    @property
    def classes_(self):
        return self._wrap_attribute('classes_')


class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects some columns from the input DataFrame.
    If columns are missing, can fill them with a given value, or throw ValueError.

    Parameters
    ----------
    columns : hashable or iterable of hashables, optional
        Column or columns to select.
        If not provided, all columns present during `fit` are selected.

    fill_value : scalar, optional
        Value to fill for missing columns.
        If not provided, will use NaN.

    handle_missing : {"ignore", "error"}, default "ignore"
        Whether to raise a ValueError if columns are missing during `transform`.
    """
    logger = ClassLogger()

    def __init__(
            self,
            columns: Hashable | Iterable[Hashable] | None = None,
            *,
            fill_value: Scalar | None = None,
            handle_missing: Literal["ignore", "error"] = "ignore",
    ):
        self.columns = columns
        self.fill_value = fill_value
        self.handle_missing = handle_missing

    def fit(self, X, y=None):
        X, _ = check_dataframe(X)
        if self.columns is None:
            self.features_ = X.columns
        else:
            self.features_ = to_list(self.columns) if is_iterable(self.columns) else self.columns
            if diff := set(to_list(self.features_)).difference(X):
                self.logger.warning(f"Columns {diff} do not exist in dataframe.")
        return self

    def transform(self, X):
        check_is_fitted(self, 'features_')
        if self.handle_missing not in ("ignore", "error"):
            raise ValueError(
                f"Invalid value '{self.handle_missing}' given for `handle_missing`. "
                "Use 'ignore' or 'error'."
            )
        X, _ = check_dataframe(X)
        if self.handle_missing == "error" and (missing := set(to_list(self.features_)).difference(X)):
            raise ValueError(f"Columns {missing} are missing from dataframe.")
        if is_iterable(self.features_):
            Xt = X.reindex(columns=self.features_, copy=False, fill_value=self.fill_value)
        elif self.features_ in X:
            Xt = X[self.features_]
        else:
            Xt = pd.Series(index=X.index, data=self.fill_value, name=self.features_)
        return Xt

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'features_')
        return np.asarray(to_list(self.features_))


class FeatureGenerator(BaseEstimator, TransformerMixin, Generic[KT, DT, VT]):
    """
    Generate a DataFrame of features from an iterable of input samples.
    
    This is a sklearn-API wrap around `epic.pandas.create.df_from_iterable`.

    Parameters
    ----------
    feature_making_func : callable, optional
        A function for making features from each sample (sample -> dict[columns_name, value]).
        If the sample is a dict, will be applied *after* each sub-dict (if exists) has been collapsed.
        If the function returns None, this sample will be skipped.

    collapse_subdict : {'multilevel', 'join', None}, default None
        How to treat sub-dictionaries.
        'multilevel': Create a tuple (key, subkey) item for each subkey and add to the main dict.
                      If all keys are such tuples, a multilevel index is created for the DataFrame.
        'join':       Create a key_subkey item for each subkey.
        None:         Do nothing.

    keyfunc : callable, optional
        A function creating a key from each sample.
        Must be provided if the input to `transform` is an iterable yields samples and not (key, sample) pairs.
        If transforming a single sample and `keyfunc` is not provided, a dummy key will be used.

    sparse_values : dict, optional
        Specifies the columns that should be sparse, and maps them to their sparse fill values.

    n_workers : int or float, default 1
        Number of workers (subprocesses or threads) to use.
        If 0 or 1, no parallelization is performed.
        When using the "threading" backend, a positive int must be provided.
        See `ultima.Workforce` for more details.

    backend : 'multiprocessing', 'threading', 'inline' or other options, default 'multiprocessing'
        The backend to use for parallelization.
        See `ultima.Workforce` for more details.

    See Also
    --------
    epic.pandas.create.df_from_iterable : The function used internally here.
    """
    DUMMY_KEY = 'dummy_key'

    def __init__(
            self,
            feature_making_func: Callable[[DT], dict[Hashable, VT] | None] | None = None,
            *,
            collapse_subdict: Literal['multilevel', 'join', None] = None,
            keyfunc: Callable[[DT], KT] | None = None,
            sparse_values: Mapping[Hashable, VT] | None = None,
            n_workers: int | float = 1,
            backend: BackendArgument = "multiprocessing",
    ):
        self.feature_making_func = feature_making_func
        self.collapse_subdict = collapse_subdict
        self.keyfunc = keyfunc
        self.sparse_values = sparse_values
        self.n_workers = n_workers
        self.backend = backend

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[tuple[KT, DT]] | Iterable[DT] | DT) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : iterable or a single sample
            If `X` is an iterable, must yield either (key, sample) pairs, or just samples.
            Otherwise, it is a single sample.

        Returns
        -------
        DataFrame, shape (n_samples, n_features)
            If a single sample was provided and no `keyfunc` is used, its index will be a dummy key.
        """
        if not is_iterable(X) or isinstance(X, Mapping):
            if self.keyfunc is None:
                X = [(self.DUMMY_KEY, X)]
            n_workers = 1
        else:
            n_workers = self.n_workers
        return df_from_iterable(
            iterable=X,
            collapse_subdict=self.collapse_subdict,
            transform=self.feature_making_func,
            keyfunc=self.keyfunc,
            sparse_values=self.sparse_values,
            n_workers=n_workers,
            backend=self.backend,
            ordered=True,
        )


class DataDumper(SimpleTransformer):
    """
    A transformer that dumps the data into a file.
    Otherwise, the data is passed along as is.

    Parameters
    ----------
    filename : str, optional
        Name of the file to dump the data to.
        Must be set before `fit` is called.

    See Also
    --------
    epic.pandas.utils.pddump : Dumper used. Determines format by `filename` extension.
    """
    def __init__(self, filename: str | None = None):
        self.filename = filename

    def transform(self, X):
        if self.filename is None:
            raise ValueError("Must provide a filename to DataDumper.")
        X = super().transform(X)
        pddump(X, self.filename)
        return X


class PipelineLogger(SimpleTransformer):
    """
    A transformer which emits a message to a log.
    Otherwise, the data is passed along as is.

    Parameters
    ----------
    message : string, default ''
        Message to emit.

    level : int or string, default 'INFO'
        Level of the log message.

    name : string, default 'PipelineLogger'
        Name of the logger to use.
        If a logger with this name already exists, no new logger will be created.

    handlers : iterable of Handle objects, default ()
        Handlers to add to the logger.
        If any are provided, any handlers already attached to the logger (if it already exists) are removed first.

    See Also
    --------
    epic.logging.get_logger : Function used to get or create the logger.
    """
    def __init__(
            self,
            message: str = '',
            *,
            level: str | int = 'INFO',
            name: str = 'PipelineLogger',
            handlers: Iterable[Handler] = (),
    ):
        self.message = message
        self.level = level
        self.name = name
        self.handlers = handlers

    def transform(self, X):
        X = super().transform(X)
        logger = get_logger(self.name, self.level, *self.handlers)
        logger.log(logger.level, self.message)
        if hasattr(X, 'shape'):
            logger.log(logger.level, f"  shape is {X.shape}")
        elif hasattr(X, '__len__'):
            logger.log(logger.level, f"  length is {len(X)}")
        return X


class ParallelFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer similar to `sklearn.preprocessing.FunctionTransformer`.
    If maps a function over an iterable input, but does it in parallel.

    Parameters
    ----------
    func : callable, optional
        The function to map over the samples.
        If not provided, the identity function is used.

    n_workers : int or float, default -1
        Number of workers (subprocesses or threads) to use.
        If 0 or 1, no parallelization is performed.
        When using the "threading" backend, a positive int must be provided.
        See `ultima.Workforce` for more details.

    backend : 'multiprocessing', 'threading', 'inline' or other options, default 'multiprocessing'
        The backend to use for parallelization.
        See `ultima.Workforce` for more details.

    ordered : bool, default True
        Whether the order of the inputs should be kept.

    skip_false : bool, default True
        If True, any output of `func` which is False is not yielded.

    kwargs : mapping, optional
        Arguments to send to `func`.

    See Also
    --------
    ultima.Workforce : Underlying mechanism used.
    """
    def __init__(
            self,
            func: Callable | None = None,
            *,
            n_workers: int | float = -1,
            backend: BackendArgument = "multiprocessing",
            ordered: bool = True,
            skip_false: bool = True,
            kwargs: Mapping[str, Any] | None = None,
    ):
        self.func = func
        self.n_workers = n_workers
        self.backend = backend
        self.ordered = ordered
        self.skip_false = skip_false
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, iterable):
        if self.func is None:
            yield from iterable
            return
        func = partial(self.func, **self.kwargs) if self.kwargs is not None else self.func
        with Workforce(self.backend, self.n_workers if self.n_workers != 1 else 0) as workforce:
            for x in workforce.map(func, iterable, ordered=self.ordered):
                if x or not self.skip_false:
                    yield x
