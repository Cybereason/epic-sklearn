import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from epic.common.general import coalesce
from sklearn.utils import check_consistent_length
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


def recall_given_precision_score(
        y_true: ArrayLike,
        probas_pred: ArrayLike,
        precision: float,
        *,
        pos_label: int | str | None = None,
        sample_weight: ArrayLike | None = None,
) -> float:
    """
    Calculate the recall which yields the minimal precision not lower than a given value.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
        If labels are neither in {-1, 1} nor in {0, 1}, then pos_label should be provided.

    probas_pred : array-like
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    precision : float
        Precision value for which the recall is required.

    pos_label : int or str, optional
        The label of the positive class.
        If not provided and `y_true` labels are in either {-1, 1} or {0, 1}, then set to 1.

    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    float
        Recall value corresponding to `precision`.
    """
    prc, rec, _ = precision_recall_curve(y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight)
    diff = prc - precision
    return rec[np.where(diff < 0, 1.1, diff).argmin()]


def confusion_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        price: ArrayLike | pd.DataFrame | None = None,
        classprobs: ArrayLike | pd.Series | None = None,
        *,
        sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Calculate the confusion score between two sets of predictions.

    The score takes into account the probability for each class to occur (on the true labels),
    as well as the price for each classification mistake:

    .. math:: score = \sum_{ij} C_{ij} P_{ij} p_i / s_i

    where
        - C is the confusion matrix (occurrence of each pair of classes),
        - P is the price matrix,
        - p is the class probability (for the true labels),
        - s is the class support (for the true labels).

    Usually, the price matrix is positive on the diagonal (reward points) and negative
    elsewhere (price for each type of classification mistake).

    Parameters
    ----------
    y_true : array-like
        True labels.

    y_pred : array-like
        Predicted labels.

    price : array-like or DataFrame, optional
        Pricing matrix, with shape (n_classes_in_true, n_classes_in_pred).
        If an array is given, the classes are inferred as those that appear at least once
        in each data set, in sorted order.
        If not provided, the pricing matrix has 1 on the diagonal and -1 elsewhere.

    classprobs : array-like or Series, optional
        Probability of appearance for each class.
        If an array is given instead of a Series, classes are assumed in the same manner
        as for `price` (using true labels). If not provided, assigns equal probabilities to
        all classes.
        Given probabilities will be normalized internally.

    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    float
        Calculated score.
    """
    check_consistent_length(y_true, y_pred, sample_weight)
    labels = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'weight': 1 if sample_weight is None else sample_weight,
    })
    if len(labels) != len(y_true):
        raise ValueError("Index mismatch for `y_true`, `y_pred` and `sample_weight`.")
    confusion = labels.groupby(['y_true', 'y_pred'])['weight'].sum().unstack(fill_value=0)
    if isinstance(price, pd.DataFrame):
        price = price.reindex(index=confusion.index, columns=confusion.columns, copy=False, fill_value=0)
    elif price is None:
        price = pd.DataFrame(
            data=np.full_like(confusion, -1),
            index=confusion.index.sort_values(),
            columns=confusion.columns.sort_values(),
        )
        diag = price.index.intersection(price.columns)
        price.values[price.index.get_indexer(diag), price.columns.get_indexer(diag)] = 1
    else:
        price = pd.DataFrame(
            data=price,
            index=confusion.index.sort_values(),
            columns=confusion.columns.sort_values(),
        ).fillna(0)
    if isinstance(classprobs, pd.Series):
        classprobs = classprobs.reindex(index=confusion.index, copy=False, fill_value=0)
    else:
        classprobs = pd.Series(coalesce(classprobs, 1), index=confusion.index.sort_values()).fillna(0)
    support = confusion.sum(axis=1)
    return (confusion * price).mul(classprobs / support, axis=0).fillna(0).values.sum() / classprobs.sum()


def recall_over_precision_goal_score(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        precision_goal: float = 1,
        coeff: float = 20,
        *,
        pos_label: int | str = 1,
        sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Calculate a recall-based score for a specific label, in a scenario where
    a precision goal is required. Being better than the precision goal wins no points,
    whereas dropping below it causes the effective recall score value to drop exponentially. 
     
    The score is calculated as:
    
    .. math:: score = r * exp(k * min(p - p_g, 0)))

    where
        - r is the recall on the positive label,
        - p is the precision on the positive label,
        - p_g is the precision goal,
        - k is the penalty coefficient for dropping beneath the goal.

    Parameters
    ----------
    y_true : array-like
        True labels.

    y_pred : array-like
        Predicted labels.

    precision_goal : float, default 1
        The precision goal.
        Should be in the range [0, 1].

    coeff : float, default 20
        A penalty coefficient for dropping below expected precision.

    pos_label : str or int, default 1
        The class to consider when calculating precision and recall.

    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    float
        Score in the range [0, 1].

    Examples
    --------
    A recall of 1.0 will always give a score of 1.0.

    >>> recall_over_precision_goal_score([0, 0, 1, 1], [0, 0, 1, 1], 0.5)
    1.0

    Similarly, a recall of 0 will always give a score of 0.

    >>> recall_over_precision_goal_score([0, 0, 1, 1], [1, 1, 0, 0], 0.5)
    0.0

    Here, the precision is 0.5, not below the goal, and so the recall of 0.5
    is kept.

    >>> recall_over_precision_goal_score([0, 0, 1, 1], [0, 1, 1, 0], 0.5)
    0.5

    However, if the precision goal is larger, the recall is attenuated.

    >>> recall_over_precision_goal_score([0, 0, 1, 1], [0, 1, 1, 0], 0.55)
    0.1839...
    """
    check_consistent_length(y_true, y_pred)
    if not 0 <= precision_goal <= 1:
        raise ValueError(f"Invalid `exp_precision`: {precision_goal}")
    [precision], [recall], _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[pos_label], sample_weight=sample_weight,
    )
    return np.exp(coeff * min(precision - precision_goal, 0)) * recall


def eta_separation_score(population1: ArrayLike, population2: ArrayLike) -> float:
    r"""
    Measure the separation between two populations, normalized by their combined STDs:

    .. math:: \eta(X, Y) = | Mean[X] - Mean[Y] | / sqrt(Var[X] + Var[Y])

    Parameters
    ----------
    population1, population2 : array-like
        Input populations.

    Returns
    -------
    float
    """
    population1 = np.asarray(population1)
    population2 = np.asarray(population2)
    return np.abs(population1.mean() - population2.mean()) / np.sqrt(population1.var() + population2.var())
