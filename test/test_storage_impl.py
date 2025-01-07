import pytest

from opendg._storage import (
    DGStorageBackends,
    get_dg_storage_backend,
    set_dg_storage_backend,
)
from opendg._storage.backends import DGStorageDictBackend


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


def test_init(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2
    assert storage.time_granularity == 4


def test_init_from_events(DGStorageImpl):
    events = [(1, 2, 3), (5, 10, 20)]
    storage = DGStorageImpl.from_events(events)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2
    assert storage.time_granularity == 4


def test_slice_time(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_time(1, 2)
    assert storage.to_events() == [(1, 2, 3)]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 2
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_granularity == -1


def test_slice_time_bad_slice(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    with pytest.raises(ValueError):
        storage.slice_time(2, 1)


def test_slice_nodes(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_nodes([1, 2])
    assert storage.to_events() == [(1, 2, 3)]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 2
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_granularity == -1


def test_slice_nodes_empty_slice(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_nodes([])
    assert storage.to_events() == []
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_granularity == -1


def test_get_nbrs(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    nbrs = storage.get_nbrs([0, 2, 20])
    assert nbrs == {2: [(3, 1)], 20: [(10, 5)]}


def test_get_nbrs_empty_nbrs(DGStorageImpl):
    events_dict = {1: [(2, 3)], 5: [(10, 20)]}
    storage = DGStorageImpl(events_dict)
    nbrs = storage.get_nbrs([0])
    assert nbrs == {}


def test_append_single_event(DGStorageImpl):
    events = [(1, 2, 3)]
    storage = DGStorageImpl.from_events(events)
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 2
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_granularity == -1

    storage = storage.append((5, 10, 20))
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2
    assert storage.time_granularity == 4


def test_append_multiple_events(DGStorageImpl):
    events = []
    storage = DGStorageImpl.from_events(events)
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_granularity == -1

    storage = storage.append([(1, 2, 3), (5, 10, 20)])
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2
    assert storage.time_granularity == 4


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_single_event_graph(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_causes_time_gap(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_bad_time_delta(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_bad_agg_func(DGStorageImpl):
    pass


def test_temporal_coarsening_empty_graph(DGStorageImpl):
    events = []
    storage = DGStorageImpl.from_events(events)
    with pytest.raises(ValueError):
        storage.temporal_coarsening(10)


def test_get_dg_storage_backend():
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_class():
    set_dg_storage_backend(DGStorageDictBackend)
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_str():
    set_dg_storage_backend('DictionaryBackend')
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_bad_str():
    with pytest.raises(ValueError):
        set_dg_storage_backend('foo')
