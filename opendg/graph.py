from __future__ import annotations

import pathlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Set, Tuple

import pandas as pd
from torch import Tensor

from opendg._io import read_events
from opendg._storage import DGSliceTracker, DGStorage
from opendg.events import Event
from opendg.timedelta import TimeDeltaDG


class DGraph:
    r"""Dynamic Graph Object provides a 'view' over a DGStorage backend."""

    def __init__(
        self,
        data: DGStorage | List[Event] | str | pathlib.Path | pd.DataFrame,
        time_delta: TimeDeltaDG = TimeDeltaDG(unit='r'),
        **kwargs: Any,
    ) -> None:
        if not isinstance(time_delta, TimeDeltaDG):
            raise ValueError(f'bad time_delta type: {type(time_delta)}')
        self.time_delta = time_delta

        if isinstance(data, DGStorage):
            self._storage = data
        else:
            events = data if isinstance(data, list) else read_events(data, **kwargs)
            if not len(events):
                raise ValueError('Tried to initialize a DGraph with empty events list')
            self._storage = DGStorage(events)

        self._slice = DGSliceTracker()

    def to_events(self) -> List[Event]:
        return self._storage.to_events(self._slice)

    def materialize(self, materialize_features: bool = True) -> DGBatch:
        r"""Materialize dense tensors: src, dst, time, and optionally {'node': node_features, 'edge': edge_features}."""
        batch = DGBatch(*self.edges)
        if materialize_features and self.node_feats is not None:
            batch.node_feats = self.node_feats._values()
        if materialize_features and self.edge_feats is not None:
            batch.edge_feats = self.edge_feats._values()
        return batch

    def slice_events(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> DGraph:
        r"""Create and return a new view by slicing events (end_idx inclusive)."""
        if start_idx is not None and end_idx is not None and start_idx > end_idx:
            raise ValueError(f'start_idx ({start_idx}) must be <= end_idx ({end_idx})')

        dg = DGraph(data=self._storage, time_delta=self.time_delta)
        dg._slice.start_time = self._slice.start_time
        dg._slice.end_time = self._slice.end_time
        dg._slice.start_idx = self._maybe_max(start_idx, self._slice.start_idx)
        dg._slice.end_idx = self._maybe_min(end_idx, self._slice.end_idx)
        dg._slice.node_slice = self._slice.node_slice
        return dg

    def slice_time(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> DGraph:
        r"""Create and return a new view by slicing temporally (end_time inclusive)."""
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f'start_time ({start_time}) must be <= end_time ({end_time})'
            )

        dg = DGraph(data=self._storage, time_delta=self.time_delta)
        dg._slice.start_time = self._maybe_max(start_time, self.start_time)
        dg._slice.end_time = self._maybe_min(end_time, self.end_time)
        dg._slice.start_idx = self._slice.start_idx
        dg._slice.end_idx = self._slice.end_idx
        dg._slice.node_slice = self._slice.node_slice
        return dg

    def slice_nodes(self, nodes: List[int]) -> DGraph:
        r"""Create and return a new view by slicing nodes to include."""
        dg = DGraph(data=self._storage, time_delta=self.time_delta)

        # Take intersection of nodes
        if self._slice.node_slice is None:
            self._slice.node_slice = set(range(self.num_nodes))
        dg._slice.node_slice = self._slice.node_slice & set(nodes)

        # Update start time
        new_start_time = self._storage.get_start_time(dg._slice)
        dg._slice.start_time = self._maybe_max(new_start_time, self.start_time)

        # Update end time
        new_end_time = self._storage.get_end_time(dg._slice)
        dg._slice.end_time = self._maybe_min(new_end_time, self.end_time)

        dg._slice.start_idx = self._slice.start_idx
        dg._slice.end_idx = self._slice.end_idx
        return dg

    def __len__(self) -> int:
        r"""The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta})'

    @cached_property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._slice.start_time is None:
            self._slice.start_time = self._storage.get_start_time(self._slice)
        return self._slice.start_time

    @cached_property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._slice.end_time is None:
            self._slice.end_time = self._storage.get_end_time(self._slice)
        return self._slice.end_time

    @cached_property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        self._slice.node_slice = self._storage.get_nodes(self._slice)
        return max(self._slice.node_slice) + 1 if len(self._slice.node_slice) else 0

    @cached_property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        src, *_ = self.edges
        return len(src)

    @cached_property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.get_num_timestamps(self._slice)

    @cached_property
    def num_events(self) -> int:
        r"""The total number of events encountered over the dynamic graph."""
        return self._storage.get_num_events(self._slice)

    @cached_property
    def nodes(self) -> Set[int]:
        r"""The set of node ids over the dynamic graph."""
        return self._storage.get_nodes(self._slice)

    @cached_property
    def edges(self) -> Tuple[Tensor, Tensor, Tensor]:
        r"""The src, dst, time tensors over the dynamic graph."""
        return self._storage.get_edges(self._slice)

    @cached_property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        If node features exist, returns a Tensor.sparse_coo_tensor(T x V x d_edge).
        """
        return self._storage.get_node_feats(self._slice)

    @cached_property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        If edge features exist, returns a Tensor.sparse_coo_tensor(T x V x V x d_edge).
        """
        return self._storage.get_edge_feats(self._slice)

    @cached_property
    def node_feats_dim(self) -> Optional[int]:
        r"""Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_node_feats_dim()

    @cached_property
    def edge_feats_dim(self) -> Optional[int]:
        r"""Edge feature dimension or None if not Node features on the Graph."""
        return self._storage.get_edge_feats_dim()

    @staticmethod
    def _maybe_max(a: Any, b: Any) -> Optional[int]:
        if a is not None and b is not None:
            return max(a, b)
        return a if b is None else b if a is None else None

    @staticmethod
    def _maybe_min(a: Any, b: Any) -> Optional[int]:
        if a is not None and b is not None:
            return min(a, b)
        return a if b is None else b if a is None else None


@dataclass
class DGBatch:
    src: Tensor
    dst: Tensor
    time: Tensor
    node_feats: Optional[Tensor] = None
    edge_feats: Optional[Tensor] = None
