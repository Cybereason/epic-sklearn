import numpy as np
import pandas as pd

from typing import Literal
from scipy import sparse as sp
from collections.abc import Iterable
from sklearn.utils import check_array
from pandas.core.generic import NDFrame
from sklearn.utils.multiclass import type_of_target
from numpy.typing import ArrayLike, DTypeLike, NDArray


def assert_categorical(array: ArrayLike, name: str = "array") -> None:
    """
    Verifies than an array contains discrete values.

    Parameters
    ----------
    array : array-like
        Input array.

    name : str, default "array"
        Displayed in error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `array` is not categorical.
    """
    if type_of_target(array) not in ('binary', 'multiclass') and array.dtype != 'category':
        raise ValueError(f"{name} must be categorical.")


def check_dataframe(obj) -> tuple[pd.DataFrame, bool]:
    """
    Converts the input to a DataFrame, if needed.

    Parameters
    ----------
    obj : object
        Input object.

    Returns
    -------
    DataFrame
        Object converted to a DataFrame.
        If `obj` is already a DaraFrame, the same object is returned.

    bool
        A flag indicating whether `obj` was already a DataFrame to begin with.
    """
    if isinstance(obj, pd.DataFrame):
        is_df = True
    else:
        is_df = False
        if sp.issparse(obj):
            obj = pd.DataFrame.sparse.from_spmatrix(obj)
        else:
            obj = pd.DataFrame(obj)
    return obj, is_df


def check_array_permissive(
        array: ArrayLike | sp.spmatrix | NDFrame,
        accept_sparse: str | bool | list[str] | tuple[str] = ('csr', 'csc', 'coo', 'dok', 'bsr', 'lil', 'dia'),
        dtype: Literal['numeric'] | DTypeLike | list[DTypeLike] | tuple[DTypeLike] | None = None,
        force_all_finite: bool | Literal['allow-nan'] = False,
        **kwargs,
) -> NDArray | sp.spmatrix:
    """
    A thin wrap around `sklearn.utils.check_array` with very permissive defaults.
    By default, allows any king of sparse matrix, non-numeric arrays and arrays containing NaNs.

    Parameters
    ----------
    array : array-like, sparse matrix or pandas object
        Input array.

    accept_sparse : str, bool or list/tuple of str, default ('csr', 'csc', 'coo', 'dok', 'bsr', 'lil', 'dia')
        Sparse formats to allow, if any.

    dtype : 'numeric', type or list/tuple of types, optional
        Dtypes to allow.

    force_all_finite : bool or 'allow-nan', default False
        Whether to raise an error on np.inf, np.nan and pd.NA in `array`.

    **kwargs :
        Sent to `sklearn.utils.check_array` as is.

    Returns
    -------
    ndarray or sparse matrix
    """
    return check_array(array, accept_sparse=accept_sparse, dtype=dtype, force_all_finite=force_all_finite, **kwargs)


def rebalance_precision(precision: float, pnr_in: float, pnr_out: float = 1) -> float:
    """
    Translate precision between datasets which have different positives-to-negatives ratios.

    Parameters
    ----------
    precision : float
        Precision on input dataset.

    pnr_in : float
        Positive-to-Negative ratio on input dataset.

    pnr_out : float
        Positive-to-Negative ratio on output dataset.

    Returns
    -------
    float
        Precision on output dataset.
    """
    return 1 / ((1 / precision - 1) * pnr_in / pnr_out + 1)


def boolean_selector(arg: bool | ArrayLike | Iterable, /, length: int) -> NDArray[np.bool_]:
    """
    Create a boolean mask for selecting indices from a 1-D array.

    Parameters
    ----------
    arg : bool, array-like or iterable
        - bool:
            Mask will be filled with this value.
        - array-like or iterable of booleans:
            Must be 1-D, of length `length`.
            Just converted into an array and returned.
        - array-like or iterable of integers:
            Must be 1-D.
            Indices of the output to be filled with True; rest are False.

    length : int
        Length of returned array.

    Returns
    -------
    ndarray, shape (length,), dtype bool

    Examples
    --------
    >>> boolean_selector(True, 3)
    array([True, True, True])

    >>> boolean_selector([True, False, True], 3)
    array([True, False, True])

    >>> boolean_selector([1, 3, 4], 6)
    array([False, True, False, True, True, False])
    """
    if isinstance(arg, bool):
        return np.full(length, arg)
    if isinstance(arg, Iterable) and not isinstance(arg, np.ndarray):
        arg = list(arg)
    arg = np.asarray(arg)
    if arg.ndim != 1:
        raise ValueError(f"Expected a 1-D array; got {arg.ndim}-D instead.")
    if np.issubdtype(arg.dtype, np.integer):
        mask = np.zeros(length, dtype=np.bool_)
        mask[arg] = True
        return mask
    if arg.size != length:
        raise ValueError(f"Length of argument should be {length}; got {len(arg)} instead.")
    if not np.issubdtype(arg.dtype, np.bool_):
        raise ValueError(f"Expected a boolean dtype; got {arg.dtype} instead.")
    return arg
