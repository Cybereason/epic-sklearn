import numpy as np

from numpy.typing import ArrayLike
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler


def kneedle(
        x: ArrayLike,
        y: ArrayLike,
        sensitivity: float = 1,
        smoothing_factor: float | None = None,
) -> list[tuple[float, float]]:
    """
    Kneedle in a Haystack - Find knee points in the curve y(x).

    It is assumed that the curve has negative concavity. If this isn't the case
    (i.e., you search for an "elbow" instead of a "knee"), transform the data
    appropriately beforehand.

    Parameters
    ----------
    x : array-like
        The x values of the curve.

    y : array-like
        The y values of the curve.
        Should have the same length as `x`.

    sensitivity : float, default 1
        Algorithm sensitivity parameter.

    smoothing_factor : float, optional
        Factor passed on to `scipy.interpolate.UnivariateSpline`.
        If not provided, len(y) / 100 is used.

    Returns
    -------
    list of 2-tuples of floats
        Points (x, y) where knees were found.

    Notes
    -----
    This is an implementation of the Kneedle algorithm for finding "knee-points"
    in 2D curves, described in [1]_.

    References
    ----------
    .. [1] V. Satopaa, J. Albrecht, D. Irwin and B. Raghavan, "Finding a "Kneedle"
       in a Haystack: Detecting Knee Points in System Behavior," 2011 31st International
       Conference on Distributed Computing Systems Workshops, Minneapolis, MN, USA, 2011,
       pp. 166-171, doi: 10.1109/ICDCSW.2011.20.
       https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf
    """
    x = np.ravel(x)
    y = np.ravel(y)
    if len(x) != len(y):
        raise ValueError("lengths of x and y do not match")
    if smoothing_factor is None:
        smoothing_factor = len(y) / 100
    spl = UnivariateSpline(x, y, s=smoothing_factor)
    Dsn = MinMaxScaler().fit_transform(np.column_stack((x, spl(x))))
    yd = np.diff(Dsn).flatten()
    delta = np.diff(yd)
    lmx_ind = np.where((delta < 0) & (np.insert(delta[:-1], 0, -1) > 0))[0]
    T = yd[lmx_ind] - sensitivity * np.diff(Dsn[:, 0]).mean()
    knees = []
    for i in range(len(lmx_ind)):
        ind_j = lmx_ind[i + 1] if i != len(lmx_ind) - 1 else len(yd)
        sl = slice(lmx_ind[i] + 1, ind_j)
        y_sl = yd[sl]
        d_sl = np.append(delta, 0)[sl]
        pos = d_sl > 0
        if pos.any():
            y_sl = y_sl[:np.where(pos)[0].min()]
        if np.any(y_sl < T[i]):
            knees.append(lmx_ind[i])
    return list(zip(x[knees], y[knees]))
