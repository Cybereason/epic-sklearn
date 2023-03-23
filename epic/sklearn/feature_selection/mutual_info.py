import warnings
import numpy as np
import pandas as pd

from typing import Literal
from functools import partial
from scipy import sparse as sp
from scipy.special import digamma
from numpy.typing import ArrayLike, NDArray

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state, check_X_y, check_array, tosequence

from ultima import Workforce
from epic.logging import get_logger
from epic.common.general import is_iterable

from ..utils import boolean_selector
from ..neighbors import NearestNeighbors

_PARALLEL_BUFFER_SIZE = 100


def _compute_cmi_xxd(x: NDArray, y: NDArray, z: NDArray, x_discrete: bool, y_discrete: bool, n_neighbors: int) -> float:
    """
    Compute conditional mutual information between x and y, given z, for a discrete z.

    .. math:: I(X;Y|Z) = E_Z[I(X;Y)|Z]

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two continuous random variables.
        Must have the same shape.

    z : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    x_discrete, y_discrete : bool
        Whether x and y are discrete or continuous.

    n_neighbors : int
        Number of nearest neighbors to search for each point.

    Returns
    -------
    float
        Estimated conditional mutual information.
    """
    cmi = n = 0
    for value in np.unique(z):
        mask = z == value
        count = np.sum(mask)
        xm = x[mask]
        ym = y[mask]
        if (
                x_discrete and not y_discrete and np.all(np.unique(xm, return_counts=True)[1] == 1) or
                y_discrete and not x_discrete and np.all(np.unique(ym, return_counts=True)[1] == 1) or
                not x_discrete and not y_discrete and count <= 1
        ):
            continue
        cmi += max(_compute_mi(xm, ym, x_discrete, y_discrete, n_neighbors=n_neighbors), 0) * count
        n += count
    if n > 0:
        cmi /= n
    return cmi


def _compute_cmi_ccc(x: NDArray, y: NDArray, z: NDArray, n_neighbors: int) -> float:
    """
    Compute conditional mutual information between x and y, given z, where all three variables are continuous.

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples,)
        Samples of three continuous random variables
        Must all have the same shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    float
        Estimated conditional mutual information.
        If it turns out to be negative, 0 is returned instead.

    References
    ----------
    .. [1] A. Tsimpiris, I. Vlachos and D. Kugiumtzis, "Nearest neighbor estimate of conditional mutual
           information in feature selection". Expert Systems with Applications 39, 2012.
    """
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    xyz = np.hstack((x, y, z))

    nn = NearestNeighbors(metric='chebyshev')
    radius = _radius_of_kth_neighbor(nn, xyz, n_neighbors)

    nxz = _n_neighbors_by_radius(nn, xyz[:, [0, 2]], radius)
    nyz = _n_neighbors_by_radius(nn, xyz[:, [1, 2]], radius)
    nz = _n_neighbors_by_radius(nn, z, radius)

    cmi = digamma(n_neighbors) + np.mean(digamma(nz + 1)) - np.mean(digamma(nxz + 1)) - np.mean(digamma(nyz + 1))
    return max(0, cmi)


def _compute_cmi_ddc(x: NDArray, y: NDArray, z: NDArray, n_neighbors: int) -> float:
    """
    Compute conditional mutual information between x and y, discrete variables, given z, a continuous variable.

    .. math:: I(X;Y|Z) = I(X;Y) + H(Z|X) + H(Z|Y) - H(Z) - H(Z|X,Y)

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two discrete random variables
        Must have the same shape.

    z : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point.

    Returns
    -------
    float
        Estimated conditional mutual information.
        If it turns out to be negative, 0 is returned instead.
    """
    n_samples = z.size
    z = z.reshape((-1, 1))

    # This is a 1-D array of pairs
    xy = np.empty(n_samples, dtype=[('x', x.dtype), ('y', y.dtype)])
    xy['x'] = x
    xy['y'] = y

    radius = np.empty(n_samples)
    nxy = np.ones(n_samples, dtype=np.uint32)
    k_all = np.empty(n_samples, dtype=np.uint8)

    nn = NearestNeighbors()

    values, counts = np.unique(xy, return_counts=True)
    mask = counts > 1
    values = values[mask]
    counts = counts[mask]

    for value, count in zip(values, counts):
        mask = xy == value
        nxy[mask] = count
        k = min(n_neighbors, count - 1)
        k_all[mask] = k
        radius[mask] = _radius_of_kth_neighbor(nn, z[mask], k)

    # Ignore points with unique (x, y) values.
    mask = nxy > 1
    n_samples = np.sum(mask)
    if n_samples == 0:
        return 0

    radius = radius[mask]
    nxy = nxy[mask]
    k_all = k_all[mask]
    x = x[mask]
    y = y[mask]
    z = z[mask]

    mx = np.empty(n_samples, dtype=np.uint32)
    nx = np.empty(n_samples, dtype=np.uint32)

    for vx in np.unique(x):
        mask = x == vx
        nx[mask] = np.sum(mask)
        mx[mask] = _n_neighbors_by_radius(nn, z[mask], radius[mask])

    my = np.empty(n_samples, dtype=np.uint32)
    ny = np.empty(n_samples, dtype=np.uint32)

    for vy in np.unique(y):
        mask = y == vy
        ny[mask] = np.sum(mask)
        my[mask] = _n_neighbors_by_radius(nn, z[mask], radius[mask])

    m = _n_neighbors_by_radius(nn, z, radius)

    cmi = (
            mutual_info_score(x, y) -
            np.mean(digamma(mx + 1)) +
            np.mean(digamma(nx)) -
            np.mean(digamma(my + 1)) +
            np.mean(digamma(ny)) +
            np.mean(digamma(m + 1)) -
            digamma(n_samples) +
            np.mean(digamma(k_all)) -
            np.mean(digamma(nxy))
    )
    return max(0, cmi)


def _compute_cmi_dcc(x: NDArray, y: NDArray, z: NDArray, n_neighbors: int) -> float:
    """
    Compute conditional mutual information between x (discrete) and y (continuous), given z, a continuous variable.

    .. math:: I(X;Y|Z) = H(Z|X) + H(Y,Z) - H(Z) - H(Y,Z|X)

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    y, z : ndarray, shape (n_samples,)
        Samples of two continuous random variables.
        Must have the same shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point.

    Returns
    -------
    float
        Estimated conditional mutual information.
        If it turns out to be negative, 0 is returned instead.
    """
    n_samples = z.size
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    yz = np.hstack((y, z))

    radius = np.empty(n_samples)
    nx = np.ones(n_samples, dtype=np.uint32)
    mzx = np.empty(n_samples, dtype=np.uint32)
    k_all = np.empty(n_samples, dtype=np.uint8)

    nn = NearestNeighbors(metric='chebyshev')

    values, counts = np.unique(x, return_counts=True)
    mask = counts > 1
    values = values[mask]
    counts = counts[mask]

    for value, count in zip(values, counts):
        mask = x == value
        nx[mask] = count
        k = min(n_neighbors, count - 1)
        k_all[mask] = k
        r = _radius_of_kth_neighbor(nn, yz[mask], k)
        radius[mask] = r
        mzx[mask] = _n_neighbors_by_radius(nn, z[mask], r)

    # Ignore points with unique x values.
    mask = nx > 1
    if not np.any(mask):
        return 0

    radius = radius[mask]
    mzx = mzx[mask]
    k_all = k_all[mask]
    z = z[mask]
    yz = yz[mask]

    mz = _n_neighbors_by_radius(nn, z, radius)
    m = _n_neighbors_by_radius(nn, yz, radius)

    cmi = np.mean(digamma(k_all)) - np.mean(digamma(m + 1)) + np.mean(digamma(mz + 1)) - np.mean(digamma(mzx + 1))
    return max(0, cmi)


def _compute_cmi(x: NDArray, y: NDArray, z: NDArray,
                 x_discrete: bool, y_discrete: bool, z_discrete: bool,
                 n_neighbors: int) -> float:
    """
    Compute the conditional mutual information between x and y, given z.

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples,)
        Samples of three random variables
        Must all have the same shape.

    x_discrete, y_discrete, z_discrete : bool
        Whether x, y and z are discrete or continuous.

    n_neighbors : int
        Number of nearest neighbors to search for each point.

    Returns
    -------
    float
        Estimated conditional mutual information.
    """
    if z_discrete:
        return _compute_cmi_xxd(x, y, z, x_discrete, y_discrete, n_neighbors)
    if x_discrete and y_discrete:
        return _compute_cmi_ddc(x, y, z, n_neighbors)
    if x_discrete and not y_discrete:
        return _compute_cmi_dcc(x, y, z, n_neighbors)
    if not x_discrete and y_discrete:
        return _compute_cmi_dcc(y, x, z, n_neighbors)
    return _compute_cmi_ccc(x, y, z, n_neighbors)


def _radius_of_kth_neighbor(nearest_neighbors: NearestNeighbors, data: NDArray, n_neighbors: int) -> float:
    nearest_neighbors.set_params(n_neighbors=n_neighbors, algorithm='auto')
    nearest_neighbors.fit(data)
    radius = nearest_neighbors.kneighbors()[0]
    return np.nextafter(radius[:, -1], 0)  # subtract epsilon


def _n_neighbors_by_radius(nearest_neighbors: NearestNeighbors, data: NDArray, radius: NDArray) -> NDArray:
    # Algorithm is selected explicitly to allow passing an array as radius,
    # as not all algorithms support this.
    nearest_neighbors.set_params(algorithm='kd_tree')
    nearest_neighbors.fit(data)
    return nearest_neighbors.radius_neighbors_count(radius=radius)


def _get_column(data: NDArray | sp.spmatrix, col: int) -> NDArray:
    if sp.issparse(data):
        # This assumes data is in CSR format.
        c = np.zeros(data.shape[0], dtype=data.dtype)
        start_ptr, end_ptr = data.indptr[[col, col + 1]]
        c[data.indices[start_ptr:end_ptr]] = data.data[start_ptr:end_ptr]
        return c
    return data[:, col]


def _preprocess_data(
        X: ArrayLike | sp.spmatrix | pd.DataFrame,
        y: ArrayLike | pd.Series,
        discrete_features: Literal['auto'] | bool | ArrayLike,
        discrete_target: bool,
        copy: bool,
        random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray | sp.spmatrix, NDArray, NDArray[np.bool_], pd.Index | None]:
    """
    Preprocess the data before computing mutual information.

    - Continuous features, and the target if continuous, are scaled, and small amount of
      noise is added, as advised in Kraskov et al.
    - If `X` is a DataFrame, any discrete features are encoded into integers using LabelEncoder.


    Parameters
    ----------
    X : array-like, sparse matrix or DataFrame, shape (n_samples, n_features)
        Feature matrix.

    y : array-like or Series, shape (n_samples,)
        Target vector.

    discrete_features : 'auto', bool or array-like
        Which of the features are discrete and which are continuous.
        - bool: All features are discrete (True) or continuous (False).
        - array-like: Either a boolean mask with shape (n_features,) indicating
                      the discrete features, an array with indices of discrete features
                      or an array of discrete feature names (if `X` is a DataFrame).
        - 'auto': Equivalent to False for dense `X` and to True for sparse `X`.

    discrete_target : bool
        Whether to consider `y` as a discrete variable.

    copy : bool
        Whether to make a copy of the given data.
        If set to False, the initial data will be overwritten.
        If `X` is a DataFrame and there are discrete features the data is copied regardless.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    Returns
    -------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        Features matrix.

    y : ndarray, shape (n_samples,)
        Target vector.

    discrete_mask : ndarray, shape (n_features,)
        Boolean mask specifying which of the features are discrete.

    feature_names : Index or None, shape (n_features,)
        Names of the features, if known.
    """
    log = get_logger()
    log.debug("preprocessing features")

    if is_iterable(discrete_features):
        discrete_features = np.asarray(tosequence(discrete_features))

    elif isinstance(discrete_features, str) and discrete_features == 'auto':
        discrete_features = sp.issparse(X)

    if isinstance(X, pd.DataFrame):
        if isinstance(y, pd.Series):
            if not X.index.equals(y.index):
                raise ValueError("Index mismatch between `X` and `y`.")
            y = y[X.index]

        if (
                isinstance(discrete_features, np.ndarray) and
                not np.issubdtype(discrete_features.dtype, np.integer) and
                not np.issubdtype(discrete_features.dtype, np.bool_)
        ):
            discrete_features = X.columns.get_indexer(discrete_features)
            if np.any(m := discrete_features == -1):
                raise ValueError(f"Features {discrete_features[m].tolist()} not found in `X`.")

        discrete_mask = boolean_selector(discrete_features, X.shape[1])

        if np.any(discrete_mask):
            log.debug("encoding discrete features as ints")
            X = X.apply(
                lambda x, dsc: LabelEncoder().fit_transform(x) if x.name in dsc else x,
                dsc=X.columns[discrete_mask],
            )
            copy = False

        feature_names = X.columns

    elif (
            isinstance(discrete_features, np.ndarray) and
            not np.issubdtype(discrete_features.dtype, np.integer) and
            not np.issubdtype(discrete_features.dtype, np.bool_)
    ):
        raise ValueError("Invalid `discrete_features` when `X` is not a DataFrame")

    else:
        feature_names = None
        discrete_mask = None

    if discrete_target:
        check_classification_targets(y)

    log.debug("checking input format")
    X, y = check_X_y(X, y, accept_sparse='csc', y_numeric=not discrete_target)
    n_samples, n_features = X.shape

    if discrete_mask is None:
        discrete_mask = boolean_selector(discrete_features, n_features)

    continuous_mask = ~discrete_mask
    rng = check_random_state(random_state)

    if np.any(continuous_mask):
        log.debug("preprocessing continuous features")

        if sp.issparse(X):
            if not np.issubdtype(X.dtype, np.floating):
                X = X.astype(np.double)  # always performs a copy
            elif copy:
                X = X.copy()
        else:
            X = X.astype(np.double, copy=copy)

        log.debug("scaling continuous features")
        continuous = scale(X[:, continuous_mask], with_mean=False, copy=False)

        # Add small noise to continuous features as advised in Kraskov et al.
        log.debug("adding small noise to continuous features")
        means = np.maximum(1, np.asarray(np.abs(continuous).mean(axis=0)))

        with warnings.catch_warnings():
            # While if X is sparse this is "inefficient", the alternatives are far less efficient...
            warnings.simplefilter('ignore', category=sp.SparseEfficiencyWarning)
            X[:, continuous_mask] = continuous + 1e-10 * means * rng.randn(n_samples, np.sum(continuous_mask))

    if not discrete_target:
        log.debug("scaling target")
        y = scale(y, with_mean=False)
        log.debug("adding small noise to target")
        y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.randn(n_samples)

    log.debug("done preprocessing")
    return X, y, discrete_mask, feature_names


def conditional_mutual_info(
        X: ArrayLike | sp.spmatrix | pd.DataFrame,
        y: ArrayLike | pd.Series,
        discrete_features: Literal['auto'] | bool | ArrayLike = 'auto',
        discrete_target: bool = False,
        n_neighbors: int = 3,
        copy: bool = True,
        random_state: int | np.random.RandomState | None = None,
        cmi_matrix: ArrayLike | pd.DataFrame | None = None,
        n_workers: int | float = -1,
) -> NDArray | pd.DataFrame:
    r"""
    Estimate conditional mutual information matrix Q between the features and the target:

    .. math::

        Q_{ii} = I(X_i; y)                    \\
        Q_{ij} = I(X_i; y | X_j)    (i != j)

    Parameters
    ----------
    X : array-like, sparse matrix or DataFrame, shape (n_samples, n_features)
        Feature matrix.

    y : array-like or Series, shape (n_samples,)
        Target vector.

    discrete_features : 'auto', bool or array-like, default 'auto'
        Which of the features are discrete and which are continuous.
        - bool: All features are discrete (True) or continuous (False).
        - array-like: Either a boolean mask with shape (n_features,) indicating
                      the discrete features, an array with indices of discrete features
                      or an array of discrete feature names (if `X` is a DataFrame).
        - 'auto': Equivalent to False for dense `X` and to True for sparse `X`.

    discrete_target : bool, default False
        Whether to consider `y` as a discrete variable.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables,
        see [1]_, [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data.
        If set to False, the initial data will be overwritten.
        If `X` is a DataFrame and there are discrete features the data is copied regardless.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    cmi_matrix : array-like or DataFrame, shape (n_features, n_features), optional
        A partially-computed conditional information matrix.
        If given, entries already computed are simply copied to result.
        Entries will only be computed wherever the matrix contains NaN.

    n_workers : int or float, default -1
        Number of CPUs to use.
        - Positive int:   Number of processes.
        - Negative int:   Refers to the number of CPUs (-1 means n_CPUs, -2 means 1 fewer, etc.)
        - Positive float: Fraction of the number of CPUs.

    Returns
    -------
    Q : ndarray or DataFrame, shape (n_features, n_features)
        Estimated conditional mutual information matrix.
        A negative value will be replaced by 0.
        A DataFrame is returned if `X` is a DataFrame.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [3] A. Tsimpiris, I. Vlachos and D. Kugiumtzis, "Nearest neighbor estimate of conditional mutual
           information in feature selection". Expert Systems with Applications 39, 2012.
    """
    X, y, discrete_mask, feature_names = _preprocess_data(
        X, y, discrete_features, discrete_target, copy, random_state,
    )
    log = get_logger()
    n_features = X.shape[1]

    if isinstance(cmi_matrix, pd.DataFrame):
        if feature_names is None:
            raise ValueError("`cmi_matrix` cannot be a DataFrame when `X` is not.")
        cmi_matrix = cmi_matrix.reindex(index=feature_names, columns=feature_names)
    else:
        if cmi_matrix is None:
            cmi_matrix = np.full((n_features, n_features), np.NaN)
        else:
            cmi_matrix = check_array(cmi_matrix, force_all_finite=False)
        cmi_matrix = pd.DataFrame(cmi_matrix, index=feature_names, columns=feature_names)

    stacked = cmi_matrix.stack(dropna=False)
    to_calc = stacked.index[stacked.isnull()]
    pairs = np.column_stack((
        cmi_matrix.index.get_indexer(to_calc.get_level_values(0)),
        cmi_matrix.columns.get_indexer(to_calc.get_level_values(1)),
    ))
    diag = pairs[:, 0] == pairs[:, 1]

    with Workforce(n_workers=n_workers) as workforce:
        stacked.loc[to_calc[diag]] = np.maximum(list(workforce.map(
            func=partial(_compute_mi, y_discrete=discrete_target, n_neighbors=n_neighbors),
            inputs=((_get_column(X, i), y, discrete_mask[i]) for i in pairs[diag, 0]),
            ordered=True,
            buffering=_PARALLEL_BUFFER_SIZE,
        )), 0)

        stacked.loc[to_calc[~diag]] = list(workforce.map(
            func=partial(_compute_cmi, n_neighbors=n_neighbors),
            inputs=(
                (_get_column(X, i), y, _get_column(X, j), discrete_mask[i], discrete_target, discrete_mask[j])
                for i, j in pairs[~diag]
            ),
            ordered=True,
            buffering=_PARALLEL_BUFFER_SIZE,
        ))

    log.debug("done calculating CMI matrix elements")
    cmi_matrix = stacked.unstack()

    if feature_names is None:
        for ax in (0, 1):
            cmi_matrix.sort_index(axis=ax, inplace=True)
        cmi_matrix = cmi_matrix.values
    else:
        cmi_matrix = cmi_matrix.loc[feature_names, feature_names]

    log.debug("done building CMI matrix")
    return cmi_matrix


def spec_cmi(
        X: ArrayLike | sp.spmatrix | pd.DataFrame | None = None,
        y: ArrayLike | pd.Series | None = None,
        discrete_features: Literal['auto'] | bool | ArrayLike = 'auto',
        discrete_target: bool = False,
        n_neighbors: int = 3,
        copy: bool = True,
        random_state: int | np.random.RandomState | None = None,
        n_workers: int | float = -1,
        cmi_matrix: ArrayLike | pd.DataFrame | None = None,
) -> NDArray | pd.Series:
    """
    Rank features according to the SPEC_CMI method, described in [1]_.

    The method takes into account the mutual information between each feature and the target,
    as well as the conditional mutual information between each feature and the target,
    given every other feature.

    Parameters
    ----------
    X : array-like, sparse matrix or DataFrame, shape (n_samples, n_features), optional
        Feature matrix.

    y : array-like or Series, shape (n_samples,), optional
        Target vector.

    discrete_features : 'auto', bool or array-like, default 'auto'
        Which of the features are discrete and which are continuous.
        - bool: All features are discrete (True) or continuous (False).
        - array-like: Either a boolean mask with shape (n_features,) indicating
                      the discrete features, an array with indices of discrete features
                      or an array of discrete feature names (if `X` is a DataFrame).
        - 'auto': Equivalent to False for dense `X` and to True for sparse `X`.

    discrete_target : bool, default False
        Whether to consider `y` as a discrete variable.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables.
        Higher values reduce variance of the estimation, but could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data.
        If set to False, the initial data will be overwritten.
        If `X` is a DataFrame and there are discrete features the data is copied regardless.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    n_workers : int or float, default -1
        Number of CPUs to use.
        - Positive int:   Number of processes.
        - Negative int:   Refers to the number of CPUs (-1 means n_CPUs, -2 means 1 fewer, etc.)
        - Positive float: Fraction of the number of CPUs.

    cmi_matrix : array-like or DataFrame, shape (n_features, n_features), optional
        A pre-computed conditional information matrix.
        If given, all other parameters are ignored completely, and it is used instead
        for calculation of the weights.

    Returns
    -------
    weights : ndarray or Series, shape (n_features,)
        Relative importance of each feature.
        If either `X` or `cmi_matrix` is a DataFrames, a Series is returned with the feature names as index.

    References
    ----------
    .. [1] N. X. Vinh, J. Chan, S. Romano and J. Bailey, "Effective Global Approaches
           for Mutual Information Based Feature Selection". Proceedings of the 20th ACM
           SIGKDD international conference on Knowledge discovery and data mining (KDD '14).
    """
    if cmi_matrix is None:
        if X is None or y is None:
            raise ValueError("Must provide both `X` and `y` when not providing `cmi_matrix`.")

        cmi_matrix = conditional_mutual_info(
            X, y,
            discrete_features=discrete_features,
            discrete_target=discrete_target,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
            n_workers=n_workers,
        )

        feature_names = X.columns if isinstance(X, pd.DataFrame) else None

    elif isinstance(cmi_matrix, pd.DataFrame):
        feature_names = sorted(set(cmi_matrix.index).intersection(cmi_matrix.columns))
        cmi_matrix = cmi_matrix.loc[feature_names, feature_names].values

    else:
        feature_names = None
        cmi_matrix = check_array(cmi_matrix)
        if cmi_matrix.shape[0] != cmi_matrix.shape[1]:
            raise ValueError("Input matrix must be square.")

    eigvals, eigvecs = np.linalg.eigh((cmi_matrix + cmi_matrix.T) / 2)
    weights = eigvecs[:, np.argmax(eigvals)]
    weights[np.isclose(weights, 0, atol=np.finfo(float).eps)] = 0
    if np.all(weights <= 0):
        weights *= -1
    if np.any(weights < 0):
        raise ValueError("Error in eigenvector decomposition: inconsistent signs.")
    if feature_names is not None:
        weights = pd.Series(weights, index=feature_names)
    return weights


def iterative_max_min_cmi(
        X: ArrayLike | sp.spmatrix | pd.DataFrame | None = None,
        y: ArrayLike | pd.Series | None = None,
        discrete_features: Literal['auto'] | bool | ArrayLike = 'auto',
        discrete_target: bool = False,
        n_neighbors: int = 3,
        copy: bool = True,
        random_state: int | np.random.RandomState | None = None,
        n_workers: int | float = -1,
        tol: float = 0,
        cmi_matrix: ArrayLike | pd.DataFrame | None = None,
) -> list:
    r"""
    Choose features iteratively according to mutual information.
    
    First feature has maximal mutual information with target.
    Then, at each iteration, the next feature is chosen to have the maximal minimum mutual
    information given each of the features already selected:

    .. math:: argmax_{X_i \in C} min_{X_j \in S} I(X_i; y | X_j),

    where C is the set of candidate features (haven't been selected yet), S is the set of
    features already selected and I is the conditional mutual information.

    Parameters
    ----------
    X : array-like, sparse matrix or DataFrame, shape (n_samples, n_features), optional
        Feature matrix.

    y : array-like or Series, shape (n_samples,), optional
        Target vector.

    discrete_features : 'auto', bool or array-like, default 'auto'
        Which of the features are discrete and which are continuous.
        - bool: All features are discrete (True) or continuous (False).
        - array-like: Either a boolean mask with shape (n_features,) indicating
                      the discrete features, an array with indices of discrete features
                      or an array of discrete feature names (if `X` is a DataFrame).
        - 'auto': Equivalent to False for dense `X` and to True for sparse `X`.

    discrete_target : bool, default False
        Whether to consider `y` as a discrete variable.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables.
        Higher values reduce variance of the estimation, but could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data.
        If set to False, the initial data will be overwritten.
        If `X` is a DataFrame and there are discrete features the data is copied regardless.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    n_workers : int or float, default -1
        Number of CPUs to use.
        - Positive int:   Number of processes.
        - Negative int:   Refers to the number of CPUs (-1 means n_CPUs, -2 means 1 fewer, etc.)
        - Positive float: Fraction of the number of CPUs.

    tol : float, default 0
        If the maximal minimum conditional mutual information is not larger than `tol`,
        the algorithm stops early and fewer features are returned.

    cmi_matrix : array-like or DataFrame, shape (n_features, n_features), optional
        A pre-computed conditional information matrix.
        If given, all other parameters except `tol` are ignored, and it is used instead.

    Returns
    -------
    features : list, length n_features or shorter
        Ordering of features. First is most important, and so on.
        If either `X` or `cmi_matrix` is a DataFrames, feature names are returned.
        Otherwise, feature indices are returned.
    """
    log = get_logger()
    if cmi_matrix is None:
        if X is None or y is None:
            raise ValueError("Must provide both `X` and `y` when not providing `cmi_matrix`.")

        X, y, discrete_mask, feature_names = _preprocess_data(
            X, y, discrete_features, discrete_target, copy, random_state,
        )

        n_features = X.shape[1]
        cmi_matrix = pd.DataFrame(np.full((n_features, n_features), np.NaN), index=feature_names, columns=feature_names)

        with Workforce(n_workers=n_workers) as workforce:
            mi = list(workforce.map(
                func=partial(_compute_mi, y_discrete=discrete_target, n_neighbors=n_neighbors),
                inputs=((_get_column(X, i), y, discrete_mask[i]) for i in range(n_features)),
                ordered=True,
                buffering=_PARALLEL_BUFFER_SIZE,
            ))

            features = [cmi_matrix.columns[np.argmax(mi)]]
            log.debug(f"appending feature: {features[0]}")

            while len(features) < n_features:
                pairs = cmi_matrix.drop(features).loc[:, features].stack(dropna=False)
                pairs = pairs.index[pairs.isnull()]
                pairs = np.column_stack((
                    cmi_matrix.index.get_indexer(pairs.get_level_values(0)),
                    cmi_matrix.columns.get_indexer(pairs.get_level_values(1)),
                ))

                cmi_matrix.values[tuple(pairs.T)] = list(workforce.map(
                    func=partial(_compute_cmi, n_neighbors=n_neighbors),
                    inputs=(
                        (_get_column(X, i), y, _get_column(X, j), discrete_mask[i], discrete_target, discrete_mask[j])
                        for i, j in pairs
                    ),
                    ordered=True,
                    buffering=_PARALLEL_BUFFER_SIZE,
                ))

                min_cmi = cmi_matrix.drop(features).loc[:, features].min(axis=1)
                best = min_cmi.idxmax()
                best_cmi = min_cmi[best]
                if best_cmi <= tol:
                    break
                log.debug(f"appending feature: {best} | cmi = {best_cmi}")
                features.append(best)

    else:
        if isinstance(cmi_matrix, pd.DataFrame):
            feature_names = sorted(set(cmi_matrix.index).intersection(cmi_matrix.columns))
            cmi_matrix = cmi_matrix.loc[feature_names, feature_names]
        else:
            cmi_matrix = pd.DataFrame(check_array(cmi_matrix))

            if cmi_matrix.shape[0] != cmi_matrix.shape[1]:
                raise ValueError("Input matrix must be square.")

        features = [cmi_matrix.index[cmi_matrix.values.diagonal().argmax()]]

        while len(features) < cmi_matrix.shape[1]:
            min_cmi = cmi_matrix.drop(features).loc[:, features].min(axis=1)
            best = min_cmi.idxmax()
            best_cmi = min_cmi[best]
            if best_cmi <= tol:
                break
            log.debug(f"appending feature: {best} | cmi = {best_cmi}")
            features.append(best)

    return features
