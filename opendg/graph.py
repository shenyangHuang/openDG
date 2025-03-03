import copy
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from opendg._io import read_csv, write_csv
from opendg._storage import DGStorage, DGStorageBase
from opendg.events import Event
from opendg.timedelta import TimeDeltaDG


class DGraph:
    r"""The Dynamic Graph Object. Provides a 'view' over an internal DGStorage backend.

    Args:
        events (List[event]): The list of temporal events (node/edge events) that define the dynamic graph.
        time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
            If None, then the events are assumed 'ordered', with no specific time unit.
    """

    def __init__(
        self,
        events: Optional[List[Event]] = None,
        time_delta: Optional[TimeDeltaDG] = None,
        _storage: Optional[DGStorageBase] = None,
    ) -> None:
        if _storage is not None:
            if events is not None or time_delta is not None:
                raise ValueError(
                    'Cannot simultaneously initialize a DGraph with _storage and events/time_delta.'
                )

            self._storage = _storage
        else:
            events_list = [] if events is None else events
            self._storage = DGStorage(events_list)

        if time_delta is None:
            self._time_delta = TimeDeltaDG('r')  # Default to ordered granularity
        else:
            self._time_delta = time_delta

        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        time_delta: Optional[TimeDeltaDG] = None,
        *args: Any,
        **kwargs: Any,
    ) -> 'DGraph':
        r"""Load a Dynamic Graph from a csv_file.

        Args:
            file_path (str): The os.pathlike object to read from.
            time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
                If None, then the events are assumed 'ordered', with no specific time unit.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
           DGraph: The newly constructed dynamic graph.
        """
        events = read_csv(file_path, *args, **kwargs)
        return cls(events, time_delta)

    def to_csv(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        r"""Write a Dynamic Graph to a csv_file.

        Args:
            file_path (str): The os.pathlike object to write to.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.
        """
        events = self._storage.to_events(
            start_time=self._cache.get('start_time'),
            end_time=self._cache.get('end_time'),
            node_slice=self._cache.get('node_slice'),
        )
        write_csv(events, file_path, *args, **kwargs)

    def slice_time(self, start_time: int, end_time: int) -> 'DGraph':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        Args:
            start_time (int): The start of the temporal slice.
            end_time (int): The end of the temporal slice (exclusive).

        Returns:
            DGraph view of events between start and end_time.
        """
        self._check_slice_time_args(start_time, end_time)

        dg = copy.copy(self)
        new_start_time, new_end_time = None, None

        if self.start_time is not None and start_time > self.start_time:
            new_start_time = start_time

        if self.end_time is not None and end_time < self.end_time:
            new_end_time = end_time

        if new_start_time is not None or new_end_time is not None:
            dg._cache.clear()  # Force cache refresh
            dg._cache['start_time'] = new_start_time
            dg._cache['end_time'] = new_end_time

        return dg

    def slice_nodes(self, nodes: List[int]) -> 'DGraph':
        r"""Extract topological slice of the dynamcic graph given the list of nodes.

        Args:
            nodes (List[int]): The list of node ids to slice from.

        Returns:
            DGraph copy of events related to the input nodes.
        """
        dg = copy.copy(self)
        dg._cache.clear()

        if self._cache.get('node_slice') is None:
            self._cache['node_slice'] = set(range(self.num_nodes))

        dg._cache['node_slice'] = self._cache['node_slice'] & set(nodes)

        return dg

    def append(self, events: Union[Event, List[Event]]) -> None:
        r"""Append events to the temporal end of the dynamic graph.

        Args:
            events (Union[Event, List[Event]]): The event of list of events to add to the temporal graph.

        """
        raise NotImplementedError

    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> None:
        r"""Re-index the temporal axis of the dynamic graph.

        Args:
            time_delta (TimeDeltaDG): The time granularity to use.
            agg_func (Union[str, Callable]): The aggregation / reduction function to apply.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return f'Dynamic Graph Storage Engine ({self._storage.__class__.__name__}), Start Time: {self.start_time}, End Time: {self.end_time}, Nodes: {self.num_nodes}, Edges: {self.num_edges}, Timestamps: {self.num_timestamps}, Time Delta: {self.time_delta}'

    @property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._cache.get('start_time') is None:
            self._cache['start_time'] = self._storage.get_start_time(
                self._cache.get('node_slice')
            )
        return self._cache['start_time']

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._cache.get('end_time') is None:
            self._cache['end_time'] = self._storage.get_end_time(
                self._cache.get('node_slice')
            )
        return self._cache['end_time']

    @property
    def time_delta(self) -> TimeDeltaDG:
        r"""The time granularity of the dynamic graph."""
        return self._time_delta

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        if self._cache.get('num_nodes') is None:
            if self._cache.get('node_slice') is None:
                self._cache['node_slice'] = self._storage.get_nodes(
                    self._cache.get('start_time'), self._cache.get('end_time')
                )
            self._cache['num_nodes'] = max(self._cache['node_slice']) + 1
        return self._cache['num_nodes']

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        if self._cache.get('num_edges') is None:
            self._cache['num_edges'] = self._storage.get_num_edges(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_edges']

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        if self._cache.get('num_timestamps') is None:
            self._cache['num_timestamps'] = self._storage.get_num_timestamps(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_timestamps']

    @property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Node feature dimension
        or None if there are no node features on the dynamic graph.

        """
        if self._cache.get('node_feats') is None:
            self._cache['node_feats'] = self._storage.get_node_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['node_feats']

    @property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x V x d where

        - T = Number of timestamps
        - E = Number of edges
        - d = Edge feature dimension

        or None if there are no edge features on the dynamic graph.
        """
        if self._cache.get('edge_feats') is None:
            self._cache['edge_feats_feats'] = self._storage.get_edge_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['edge_feats']

    def _check_slice_time_args(self, start_time: int, end_time: int) -> None:
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )
