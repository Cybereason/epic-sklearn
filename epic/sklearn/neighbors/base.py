import numpy as np

from numpy.typing import ArrayLike, NDArray

from sklearn.utils.validation import check_is_fitted


class RadiusNeighborsCountMixin:
    def radius_neighbors_count(self, X: ArrayLike | None = None, radius: float | None = None) -> NDArray:
        """
        Find the number of neighbors within a given radius of a point or points.

        Return the number of points from the dataset lying in a ball with size
        `radius` around the points of the query array. Points lying on the boundary
        are included in the count.

        Parameters
        ----------
        X : array-like, (n_samples, n_features), optional
            The query point or points.
            If not provided, neighbors counts of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float, optional
            Limiting distance of neighbors to return.
            If not provided, the value passed to the constructor is used.

        Returns
        -------
        count : array, shape (n_samples,)
            Number of neighbors for each point.
        """
        check_is_fitted(self)
        if self._fit_method not in ('ball_tree', 'kd_tree'):
            # Fallback to less efficient implementation
            neighbors = self.radius_neighbors(X=X, radius=radius, return_distance=False)
            return np.fromiter((x.size for x in neighbors), dtype=int, count=neighbors.size)
        if X is None:
            query_is_train = True
            X = self._fit_X
        else:
            query_is_train = False
            X = self._validate_data(X, reset=False)
        if radius is None:
            radius = self.radius
        count = self._tree.query_radius(X, radius, count_only=True)
        # If the query data is the same as the indexed data, we would like
        # to ignore the first nearest neighbor of every sample, i.e. the sample itself.
        return count - 1 if query_is_train else count
