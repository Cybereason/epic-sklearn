import pytest
import numpy as np

from epic.sklearn.utils.general import boolean_selector


def check_bool_mask(arr, expected):
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.issubdtype(arr.dtype, np.bool_)
    assert np.all(arr == np.asarray(expected))


def test_boolean_selector():
    check_bool_mask(boolean_selector(True, 3), [True] * 3)
    check_bool_mask(boolean_selector([True, False, True], 3), [True, False, True])
    check_bool_mask(boolean_selector([1, 3, 4], 6), [False, True, False, True, True, False])
    with pytest.raises(ValueError):
        boolean_selector([[1, 2], [3, 4]], 6)
    with pytest.raises(ValueError):
        boolean_selector([True, False, True], 4)
    with pytest.raises(ValueError):
        boolean_selector(['a', 'b'], 2)
