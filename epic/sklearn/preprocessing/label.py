import numpy as np
import pandas as pd

from cytoolz import compose
from numbers import Integral

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from epic.pandas.numpy import isnan
from epic.common.general import to_list


class LabelBinarizerWithMissingValues(LabelBinarizer):
    """
    A LabelBinarizer subclass which allows for missing values.
    Missing values will result in a vector of zeros.
    """
    DEFAULT = 'NaN'

    def __init__(self, neg_label=0, pos_label=1, sparse_output=False, missing_values=DEFAULT):
        super().__init__(
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output,
        )
        self.missing_values = missing_values

    def fit(self, y):
        y = np.asanyarray(y)
        if self.missing_values == self.DEFAULT:
            valid = ~isnan(y)
        else:
            valid = y != self.missing_values
        return super().fit(y[valid])

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        if self.missing_values == self.DEFAULT:
            y = np.array(y, copy=True)
            y[isnan(y)] = self._newval()
        return super().transform(y)

    def _newval(self):
        m = max(self.classes_)
        if isinstance(m, Integral):
            return m + 1
        val = self.DEFAULT
        i = -1
        while val in self.classes_:
            i += 1
            val = f"{self.DEFAULT}{i}"
        return val


class MultiLabelEncoder(BaseEstimator):
    """
    A label encoder which allows for multiple labels for each sample.

    Each label is encoded as an integer, resulting in a list of integers for each sample,
    matching the input list of labels. Each sample can be a different number of labels.

    Parameters
    ----------
    allow_singles : bool, default False
        If True, each sample can also be a single label, which will be converted
        to a list of length 1.
    """
    def __init__(self, allow_singles: bool = False):
        self.allow_singles = allow_singles

    def _to_list(self, func):
        return compose(func, to_list) if self.allow_singles else func

    def _unique(self, y):
        classes = set()
        y.map(self._to_list(classes.update), na_action='ignore')
        return classes

    def _classmap(self, y):
        return {x: i for i, x in enumerate(sorted(self._unique(y)))}

    def fit(self, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        self.classes_ = self._classmap(y)
        return self

    def _transform(self, y, fit):
        to_numpy = False
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            to_numpy = True
        if fit:
            self.classes_ = self._classmap(y)
        elif diff := self._unique(y).difference(self.classes_):
            raise ValueError(f"`y` contains new labels: {list(diff)}.")
        yt = y.map(self._to_list(lambda v: [self.classes_[x] for x in v]), na_action='ignore')
        return yt.values if to_numpy else yt

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        return self._transform(y, fit=False)

    def fit_transform(self, y):
        return self._transform(y, fit=True)

    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')
        to_numpy = False
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            to_numpy = True
        if y.map(np.max, na_action='ignore').max() > len(self.classes_):
            raise ValueError("`y` contains new labels.")
        inv = sorted(self.classes_)
        yt = y.map(lambda v: [inv[x] for x in v], na_action='ignore')
        return yt.values if to_numpy else yt
