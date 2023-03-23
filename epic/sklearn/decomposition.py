import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_array


class TruncatedSVDOptimized(TruncatedSVD):
    """
    A transformer identical to TruncatedSVD in functionality.

    The original TruncatedSVD is optimized for inverse transformation,
    copying the components matrix each time the forward transformation is called.
    This transformer is optimized for the forward transformation,
    performing the copy in the inverse transformation instead.
    """
    def fit_transform(self, X, y=None):
        Xt = super().fit_transform(X, y)
        self.features2components_ = self.components_.T.copy()
        del self.components_
        return Xt

    def transform(self, X):
        check_is_fitted(self, 'features2components_')
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.features2components_)

    def inverse_transform(self, X):
        check_is_fitted(self, 'features2components_')
        X = check_array(X)
        return np.dot(X, self.features2components_.T)
