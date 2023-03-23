import numpy as np
import pandas as pd

from typing import Literal
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from cytoolz.curried import get, compose
from numpy.typing import ArrayLike, NDArray

from ..metrics.ranking import precision_recall_curve


class Thresholder:
    """
    A class for setting a threshold on classification probabilities, based on the value of
    a certain metric. Once the threshold is set, it can then be applied on new probabilities,
    classifying them into positive or negative labels.

    Possible metrics are:
        - fpr (false positive rate)
        - tpr (true positive rate)
        - precision
        - recall

    Parameters
    ----------
    pos_label: int or string, default 1
        Label of the positive values.

    neg_label : int or string, default 0
        Label of the negative values.

    threshold : float, default 0.5
        Known threshold. Can be readily applied without having to use `set`.
        If `set` is used, this threshold will be overwritten.
    """
    METRICS = {
        'fpr':       compose(get([0, 2]), roc_curve),
        'tpr':       compose(get([1, 2]), roc_curve),
        'precision': compose(get([0, 2]), precision_recall_curve),
        'recall':    compose(get([1, 2]), precision_recall_curve),
    }

    def __init__(self, pos_label: int | str = 1, neg_label: int | str = 0, threshold: float = 0.5):
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.threshold = threshold

    def apply(self, probas: ArrayLike) -> NDArray | pd.Series:
        """
        Apply the threshold over the probabilities.

        Parameters
        ----------
        probas : array-like
            Classification probabilities on which to apply the threshold.

        Returns
        -------
        array or Series
            Negative label where the probability was below the threshold, positive otherwise.
            If `probas` is a Series, so is the return value, with the same index.
        """
        pred = np.where(np.asarray(probas) >= self.threshold, self.pos_label, self.neg_label)
        if isinstance(probas, pd.Series):
            return pd.Series(pred, index=probas.index)
        return pred

    def set(
            self,
            y_true: ArrayLike,
            probas: ArrayLike,
            metric: Literal['fpr', 'tpr', 'precision', 'recall'],
            value: float,
            sample_weight: ArrayLike | None = None,
            interp_kind: int | Literal[
                'linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous', 'next',
            ] = 'cubic',
    ) -> float:
        """
        Set a new threshold, based on the value of a metric.

        Parameters
        ----------
        y_true : array-like
            True labels.

        probas : array-like
            Classification probabilities.

        metric : {'fpr', 'tpr', 'precision', 'recall'}
            The metric to which the `value` refers.

        value : float
            Value of the metric determining the threshold.

        sample_weight : array-like, optional
            Sample weight used in calculation of the metric.

        interp_kind : int or {'linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                              'quadratic', 'cubic', 'previous', 'next'}, default 'cubic'
            Interpolation kind used to find the correct threshold for the given `value`.
            See `scipy.interpolate.interp1d` for explanations on the various kinds.

        Returns
        -------
        float
            New threshold found.
        """
        if metric not in self.METRICS:
            raise ValueError(f"Invalid metric `{metric}`; possible values: {', '.join(self.METRICS)}.")
        values, thresholds = self.METRICS[metric](y_true, probas, pos_label=self.pos_label, sample_weight=sample_weight)
        data = pd.DataFrame({'threshold': thresholds, 'value': values})
        aggregated = data.groupby('value').agg('mean')
        curve = interp1d(aggregated.index, aggregated['threshold'], kind=interp_kind)
        self.threshold = curve(value).item()
        return self.threshold
