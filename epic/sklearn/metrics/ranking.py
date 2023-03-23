import numpy as np

from cytoolz import compose
from sklearn.metrics import precision_recall_curve as _orig_pr_curve


# Same as sklearn's `precision_recall_curve`, but returned thresholds
# have length consistent with precision and recall, with 1.0 added at the end.
precision_recall_curve = compose(lambda prt: (prt[0], prt[1], np.r_[prt[2], 1]), _orig_pr_curve)
