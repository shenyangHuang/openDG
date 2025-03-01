from abc import ABC, abstractmethod

from opendg.graph import DGraph
from opendg.timedelta import TimeDeltaDG


class DGBaseLoader(ABC):
    r"""Base class iterator over a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        drop_last (bool): Set to True to drop the last incomplete batch.

    Note:
        Ordered batch_unit ('r') iterates using normal batch size semantics
            e.g. batch_size=5, batch_unit='r' -> yield 5 events at time

        Unordered batch_unit iterates uses the appropriate temporal unit
            e.g. batch_size=5, batch_unit='s' -> yield 5 seconds of data at a time

        When using the ordered batch_unit, the order of yielded events within the same timestamp
        is non-deterministic.
    """

    def __init__(
        self,
        dg: DGraph,
        batch_size: int,
        batch_unit: str = 'r',  # TODO: Define enum
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be > 0 but got {batch_size}')

        self._dg = dg  # TODO: Clone the graph
        self._batch_time_delta = TimeDeltaDG(unit=batch_unit, value=batch_size)
        self._drop_last = drop_last

        if self._dg.time_delta.is_ordered and not self._batch_time_delta.is_ordered:
            raise ValueError(
                'Cannot use non-ordered batch_unit to iterate a DGraph with ordered time_delta'
            )

        if not self._dg.time_delta.is_ordered and not self._batch_time_delta.is_ordered:
            conversion_ratio = self._batch_time_delta.convert(self._dg.time_delta)
            batch_size = int(batch_size * conversion_ratio)  # TODO: Loss of information
            self._batch_time_delta = TimeDeltaDG(unit=batch_unit, value=batch_size)

        self._batch_start_time = self._dg.start_time
        self._batch = None

    @abstractmethod
    def sample(self, batch: 'DGraph') -> 'DGraph':
        r"""Downsample a given temporal batch, using neighborhood information, for instance.

        Args:
            batch (DGraph): Incoming batch of data. May not be materialized.

        Returns:
            (DGraph): Downsampled batch of data. Must be naterialized.
        """

    def __iter__(self) -> 'DGBaseLoader':
        return self

    def __next__(self) -> 'DGraph':
        if self._done_iteration():
            raise StopIteration

        # TODO: No good way of getting the nodes in the DGraph
        # TODO: No good way of getting the next timestamps
        # TODO: Copy semantics are broken
        if not self._dg.time_delta.is_ordered and self._batch_time_delta.is_ordered:
            batch = DGraph([], time_delta=self._dg.time_delta)
            num_left = self._batch_time_delta.value - batch.num_nodes
            while num_left > 0:
                if self._batch is not None:
                    next_batch = self._batch
                else:
                    next_batch = self._dg.slice_time(
                        self._dg.start_time, self._dg.start_time + 1
                    )

                if next_batch is None:
                    if self._drop_last:
                        raise StopIteration
                    else:
                        return batch

                curr_batch = next_batch.slice_nodes(next_batch.nodes[:num_left])
                next_batch = next_batch.slice_nodes(next_batch.nodes[num_left:])
                batch.append(curr_batch)
                self._batch = next_batch

                num_left = self._batch_time_delta.value - batch.num_nodes
        else:
            assert self._batch_start_time is not None  # For mypy
            batch_end_time = self._batch_start_time + self._batch_time_delta.value
            batch = self._dg.slice_time(self._batch_start_time, batch_end_time)
            self._batch_start_time = batch_end_time
        return batch

    def _done_iteration(self) -> bool:
        if not len(self._dg):
            return True

        # For mypy
        assert self._batch_start_time is not None
        assert self._dg.end_time is not None
        return self._batch_start_time > self._dg.end_time
