from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

from .base import RadiusNeighborsCountMixin


class NearestNeighbors(_NearestNeighbors, RadiusNeighborsCountMixin):
    """A NearestNeighbors class which also includes the `radius_neighbors_count` method."""
