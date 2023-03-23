import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from typing import overload, Literal
from sklearn.metrics import confusion_matrix


@overload
def confusion_matrix_report(
        arg1: ArrayLike, arg2: ArrayLike, /, true_label_name: str = "True label",
        pred_label_name: str = "Predicted label", to_string: Literal[True] = True, **kwargs) -> str: ...
@overload
def confusion_matrix_report(
        arg1: ArrayLike, arg2: ArrayLike, /, true_label_name: str = "True label",
        pred_label_name: str = "Predicted label", *, to_string: Literal[False], **kwargs) -> pd.DataFrame: ...
@overload
def confusion_matrix_report(
        arg1: ArrayLike, arg2: ArrayLike, /, true_label_name: str = "True label",
        pred_label_name: str = "Predicted label", *, to_string: bool, **kwargs) -> str | pd.DataFrame: ...
def confusion_matrix_report(arg1, arg2, /, true_label_name="True label", pred_label_name="Predicted label",
                            to_string=True, **kwargs):
    """
    Compute a confusion matrix and convert it into a DataFrame.

    Can also convert the result into a string.
    Can also work on a pre-computed confusion matrix.

    Parameters
    ----------
    arg1, arg2 : array-like
        Either:
            confusion_matrix : 2-D array-like
                Pre-computed confusion matrix.

            labels : 1-D array-like
                 Names of the rows and columns of the matrix.

        Or:
            y_true : 1-D array-like
                True classes.

            y_pred : 1-D array-like
                Predicted classes.

    true_label_name : str, default "True label"
        Name for the true label index.

    pred_label_name : str, default "Predicted label"
        Name for the predicted label index.

    to_string : bool, default True
        Whether to return a string representation of the report,
        or the DataFrame of the confusion matrix itself.

    **kwargs :
        Sent to DataFrame.to_string as is.

    Returns
    -------
    DataFrame or str
        Confusion matrix or its string representation.
    """
    arg1 = np.asarray(arg1)
    if arg1.ndim == 2:  # form 1
        cm, labels = arg1, arg2
    elif arg1.ndim == 1:  # form 2
        y_true, y_pred = arg1, arg2
        labels = np.union1d(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        raise ValueError("illegal value for `arg1`: must be 1- or 2-dimensional")
    cm = pd.DataFrame(cm, index=pd.Index(labels, name=true_label_name), columns=pd.Index(labels, name=pred_label_name))
    if to_string:
        return cm.to_string(max_cols=cm.shape[1], **kwargs)
    return cm


