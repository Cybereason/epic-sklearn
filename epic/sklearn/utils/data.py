import numpy as np

from scipy import sparse as sp
from numpy.typing import ArrayLike
from itertools import repeat, islice
from typing import Generic, TypeVar, Protocol
from collections.abc import Iterator, Sequence
from sklearn.utils import check_consistent_length


class _Ilocable(Protocol):
    def iloc(self, item) -> Sequence:
        ...


Indexable = TypeVar('Indexable', bound=ArrayLike | sp.spmatrix | _Ilocable)
RandomSeed = int | ArrayLike | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | None


class BatchFactory(Generic[Indexable]):
    """
    An iterator that yields shuffled batches of the given arrays.

    After each complete pass over all the data (i.e. an epoch), the data are reshuffled and iteration continues.
    - The input arrays are not shuffled in place.
    - When several arrays are provided, they are all batched together, consistently, and a tuple of the batches
      if yielded in each iteration.

    Parameters
    ----------
    *arrays : array-like, sparse matrix or pandas object
        Input arrays.
        Must all have the same length or first dimension.

    batch_size : int, default 100
        Size of each batch.

    n_epochs : int, optional
        Number of complete passes over the entire data.
        If not provided, iteration never stops.

    random_seed : int, array-like, SeedSequence, BitGenerator or Generator, optional
        Random seed for reproducibility.
    """
    def __init__(
            self,
            *arrays: Indexable,
            batch_size: int = 100,
            n_epochs: int | None = None,
            random_seed: RandomSeed = None,
    ):
        if len(arrays) == 0:
            raise ValueError("At least one array required as input")
        self.arrays = []
        for arr in arrays:
            if sp.issparse(arr):
                arr = arr.tocsr()
            elif not hasattr(arr, 'iloc'):
                arr = np.asanyarray(arr)
            self.arrays.append(arr)
        check_consistent_length(*self.arrays)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random = np.random.default_rng(random_seed)

    def __iter__(self) -> Iterator[Indexable] | Iterator[tuple[Indexable, ...]]:
        index = np.arange(self.arrays[0].shape[0])
        for epoch in repeat(None) if self.n_epochs is None else repeat(None, self.n_epochs):
            self.random.shuffle(index)
            it = iter(index)
            while True:
                idx_batch = list(islice(it, self.batch_size))
                if len(idx_batch) < self.batch_size:
                    break
                arr_batch = tuple(getattr(arr, 'iloc', arr)[idx_batch] for arr in self.arrays)
                yield arr_batch if len(arr_batch) > 1 else arr_batch[0]
