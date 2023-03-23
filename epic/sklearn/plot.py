import numpy as np
import pandas as pd

from typing import Literal
from numpy.typing import ArrayLike
from collections.abc import Callable

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

from sklearn.utils import check_X_y, check_random_state
from sklearn.metrics import silhouette_samples, roc_curve, r2_score

from epic.pandas.utils import canonize_df_and_cols
from epic.pandas.matplotlib.colors import ColorSpec
from epic.pandas.matplotlib.plot import plot_2d_hist

from .metrics import precision_recall_curve


def plot_silhouette_scores(
        X: ArrayLike,
        labels: ArrayLike,
        *,
        sample: int | None = None,
        metric: str | Callable = 'euclidean',
        cmap: str | Colormap | None = 'Spectral',
        random_state: int | np.random.RandomState | None = None,
        ax: Axes | None = None,
        **kwargs,
) -> Axes:
    """
    Calculate the clustering silhouette scores for samples and plot them.

    This is a visual aid for determining if the number of clusters is correct.
    A vertical line is plotted indicating the mean silhouette score.

    Parameters
    ----------
    X : array-like
        Features array.
        If `metric` is "precomputed", this is a pairwise distances matrix.

    labels : array-like
        Clustering label for each sample.

    sample : int, optional
        Number of samples to draw, in order to speed the calculation.
        If not given, all samples are used.

    metric : string or callable, default 'euclidean'
        The metric to use.
        If `X` is a pairwise distances matrix, `metric` should be "precomputed".

    cmap : string, Colormap or None, default "Spectral"
        Colormap to use.

    random_state : int or RandomState, optional
        The seed of the pseudo random number generator.
        Only used if `sample` is given.

    ax : Axes, optional
        Axes on which to plot.

    **kwargs :
        Additional arguments sent to distance function.

    Returns
    -------
    Axes
    """
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    if sample is not None:
        indices = check_random_state(random_state).choice(X.shape[0], sample, replace=False)
        X, labels = X[indices], labels[indices]
        if metric == "precomputed":
            X = X.T[indices].T
    sil = pd.Series(silhouette_samples(X, labels, metric=metric, **kwargs))
    min_sil, max_sil = sil.min(), sil.max()
    label_x_pos = -0.025 * (max_sil - min_sil)
    grp = sil.groupby(labels)
    colors = plt.get_cmap(cmap, len(grp))
    y_from = space_between_clusters = 10
    if ax is None:
        ax = plt.axes()
    for i, (label, scores) in enumerate(grp):
        y_to = y_from + len(scores)
        color = colors(i)
        ax.fill_betweenx(
            np.arange(y_from, y_to), 0, scores.sort_values(), facecolor=color, edgecolor=color, alpha=0.75,
        )
        ax.text(label_x_pos, (y_from + y_to) / 2, str(label))
        y_from = y_to + space_between_clusters
    ax.set_xlim(min_sil, max_sil)
    ax.set_ylim(0, y_from)
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("Cluster")
    ax.axvline(x=sil.mean(), color="red", linestyle="--")
    ax.set_yticks([])
    return ax


def plot_regression_scatter(
        arg, /, *args,
        cmap: str | Colormap | None = 'viridis',
        diag_color: ColorSpec = 'r',
        corr_title: bool = True,
        r2_title: bool = True,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
):
    """
    Plot regression prediction against the true values, with the diagonal shown
    and possibly also the :math:`R^2` (coefficient of determination) regression score displayed.

    Parameters
    ----------
    arg, *args:
        Positional-only parameters.
        Either:
            dataframe : DataFrame
                Input frame.

            y_true_column_name, y_pred_column_name : hashable
                Column names for y_true and y_pred.

        or:
            y_true_data, y_pred_data : array-like
                True and predicted values themselves.

    cmap : string, Colormap or None, default "viridis"
        Colormap to use.

    diag_color : color spec, default "r"
        Color for the diagonal line.

    corr_title : bool, default True
        If True, adds a title to the plot showing the correlation coefficient of the two data series.

    r2_title : bool, default True
        Whether to add to the title the :math:`R^2` regression score.

    ax : Axes, optional
        Axes on which to plot.

    figsize : two-tuple of floats, optional
        Figure size.
        Only used if `ax` is not given.

    Returns
    -------
    Axes
    """
    df, y_true, y_pred = canonize_df_and_cols(arg, *args)
    ax = plot_2d_hist(df, y_true, y_pred, figsize=figsize, ax=ax, cmap=cmap, corr_title=corr_title)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    ax.plot(lim, lim, c=diag_color, ls='--', lw=1.5)
    for i in 'xy':
        getattr(ax, f'set_{i}lim')(*lim)
    ax.set_aspect('equal')
    if r2_title:
        title = ax.get_title()
        r2 = f'R^2 = {r2_score(df[y_true], df[y_pred]):.3g}'
        if len(title) > 2 and title[-1] == '$' and title.count('$') % 2 == 0:
            title = fr'{title[:-1]}\quad;\quad {r2}$'
        elif title:
            title += fr';$\quad {r2}$'
        else:
            title = f'${r2}$'
        ax.set_title(title)
    return ax


def plot_classification_curves(
        y_true: ArrayLike,
        probas_pred: ArrayLike,
        *,
        pos_label: int | str | None = None,
        sample_weight: ArrayLike | None = None,
        x_data: Literal['threshold', 'recall'] = 'recall',
        plot: str = 'ps',
        ax: Axes | None = None
) -> Axes:
    """
    Plot various curves describing the results of a binary classification.

    Possible data sets are precision, recall, specificity and threshold.
    The data sets are plotted against either the recall or the threshold.

    Parameters
    ----------
    y_true : array-like
        True labels.
        If labels are neither in {-1, 1} nor in {0, 1}, then pos_label should be provided.

    probas_pred : array-like
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    pos_label : int or str, optional
        The label of the positive class.
        If not provided and `y_true` labels are in either {-1, 1} or {0, 1}, then set to 1.

    sample_weight : array-like, optional
        Sample weights.

    x_data : {'threshold', 'recall'}, default 'recall'
        Which of the two data sets should be the x-axis values.

    plot : str, default 'ps'
        Any combination of the characters 'p', 'r', 's', and 't', representing
        precision, recall, specificity and threshold, respectively. These are the data
        sets to plot. If any of 'r' and 't' are present and is also the `x_data`, this data
        set would not be plotted.

    ax : Axes, optional
        Axes on which to plot.

    Returns
    -------
    Axes
    """
    if x_data not in ('threshold', 'recall'):
        raise ValueError(f"Invalid value for `x_data`: {x_data}.")
    plot = set(plot) - {x_data[0]}
    if not plot:
        raise ValueError("Nothing to plot!")
    if p := plot.difference('prts'):
        raise ValueError(f"`plot` contains invalid characters: '{''.join(p)}'.")
    if ax is None:
        ax = plt.axes()
    pos_label_text = f" (Positive label: {pos_label})" if pos_label is not None else ""
    handles = []
    if 'p' in plot:
        precision, recall, threshold = precision_recall_curve(
            y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight,
        )
        handles += ax.plot(eval(x_data), precision, label='Precision' + pos_label_text)
    if plot & set('srt'):
        fpr, recall, threshold = roc_curve(y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight)
        # Fix inconsistency in roc_curve threshold
        threshold[0] = min(threshold[0], 1)
        for label, data in zip(('Specificity', 'Recall', 'Threshold'), (1 - fpr, recall, threshold)):
            if label[0].lower() in plot:
                handles += ax.plot(eval(x_data), data, label=label + pos_label_text)
    ax.set_xlabel(x_data.capitalize() + pos_label_text)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if len(handles) == 1:
        ax.set_ylabel(handles[0].get_label())
    else:
        ax.legend()
    return ax


def plot_precision_recall_vs_threshold(
        y_true: ArrayLike,
        probas_pred: ArrayLike,
        *,
        pos_label: int | str | None = None,
        sample_weight: ArrayLike | None = None,
        precision_goal: float | None = None,
        ax: Axes | None = None,
) -> Axes:
    """
    Plot the precision and recall against the threshold applied to prediction probabilities.

    Parameters
    ----------
    y_true : array-like
        True labels.
        If labels are neither in {-1, 1} nor in {0, 1}, then pos_label should be provided.

    probas_pred : array-like
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    pos_label : int or str, optional
        The label of the positive class.
        If not provided and `y_true` labels are in either {-1, 1} or {0, 1}, then set to 1.

    sample_weight : array-like, optional
        Sample weights.

    precision_goal : float, optional
        If provided, draws a vertical line at the highest threshold yielding precision not larger
        than `precision_goal`.

    ax : Axes, optional
        Axes on which to plot.

    Returns
    -------
    Axes
    """
    ax = plot_classification_curves(
        y_true,
        probas_pred,
        pos_label=pos_label,
        sample_weight=sample_weight,
        x_data='threshold',
        plot='pr',
        ax=ax,
    )
    if precision_goal is not None:
        if not 0 < precision_goal < 1:
            raise ValueError(f"`precision_goal` should be between 0 and 1; got {precision_goal}.")
        for line in ax.lines:
            if line.get_label().startswith('Precision'):
                data = line.get_xydata()
                ind = np.argmax(data[:, 1] > precision_goal) - 1
                ax.axvline(data[ind, 0], ls='--', lw=1, c='k')
                ax.plot(*data[ind], 'go')
                break
    return ax
