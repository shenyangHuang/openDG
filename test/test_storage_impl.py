import random

import pytest
import torch

from opendg._storage import (
    DGSliceTracker,
    DGStorageBackends,
    get_dg_storage_backend,
    set_dg_storage_backend,
)
from opendg._storage.backends import DGStorageArrayBackend
from opendg.events import EdgeEvent, NodeEvent


@pytest.fixture(autouse=True)
def seed():
    random.seed(1337)


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


@pytest.fixture
def node_only_events_list():
    return [
        NodeEvent(t=1, src=2),
        NodeEvent(t=5, src=4),
        NodeEvent(t=10, src=6),
    ]


@pytest.fixture
def node_only_events_list_with_features():
    return [
        NodeEvent(t=1, src=2, features=torch.rand(5)),
        NodeEvent(t=5, src=4, features=torch.rand(5)),
        NodeEvent(t=10, src=6, features=torch.rand(5)),
    ]


@pytest.fixture
def edge_only_events_list():
    return [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=10, src=6, dst=8),
    ]


@pytest.fixture
def edge_only_events_list_with_features():
    return [
        EdgeEvent(t=1, src=2, dst=2, features=torch.rand(5)),
        EdgeEvent(t=5, src=2, dst=4, features=torch.rand(5)),
        EdgeEvent(t=10, src=6, dst=8, features=torch.rand(5)),
    ]


@pytest.fixture
def events_list_with_multi_events_per_timestamp():
    return [
        NodeEvent(t=1, src=2),
        EdgeEvent(t=1, src=2, dst=2),
        NodeEvent(t=5, src=4),
        EdgeEvent(t=5, src=2, dst=4),
        NodeEvent(t=10, src=6),
        EdgeEvent(t=20, src=1, dst=8),
    ]


@pytest.fixture
def events_list_with_features_multi_events_per_timestamp():
    return [
        NodeEvent(t=1, src=2, features=torch.rand(5)),
        EdgeEvent(t=1, src=2, dst=2, features=torch.rand(5)),
        NodeEvent(t=5, src=4, features=torch.rand(5)),
        EdgeEvent(t=5, src=2, dst=4, features=torch.rand(5)),
        NodeEvent(t=10, src=6, features=torch.rand(5)),
        EdgeEvent(t=20, src=1, dst=8, features=torch.rand(5)),
    ]


@pytest.fixture
def events_list_out_of_time_order():
    return [
        EdgeEvent(t=5, src=2, dst=4),
        NodeEvent(t=10, src=6),
        NodeEvent(t=1, src=2),
        NodeEvent(t=5, src=4),
        EdgeEvent(t=20, src=1, dst=8),
        EdgeEvent(t=1, src=2, dst=2),
    ]


def test_attempt_init_empty(DGStorageImpl):
    with pytest.raises(ValueError):
        DGStorageImpl([])


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_node_only_events_list_to_events(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.to_events(DGSliceTracker()) == events
    assert storage.to_events(DGSliceTracker(start_time=5)) == events[1:]
    assert storage.to_events(DGSliceTracker(start_time=5, end_time=9)) == [events[1]]
    assert storage.to_events(DGSliceTracker(node_slice={1, 2, 3})) == [events[0]]
    assert (
        storage.to_events(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        == []
    )


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_edge_events_list_to_events(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.to_events(DGSliceTracker()) == events
    assert storage.to_events(DGSliceTracker(start_time=5)) == events[1:]
    assert storage.to_events(DGSliceTracker(start_time=5, end_time=9)) == [events[1]]
    assert storage.to_events(DGSliceTracker(node_slice={1, 2, 3})) == events[0:2]
    assert (
        storage.to_events(
            DGSliceTracker(start_time=6, end_time=9, node_slice={1, 2, 3})
        )
        == []
    )


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_events_list_with_multi_events_per_timestamp_to_events(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.to_events(DGSliceTracker()) == events
    assert storage.to_events(DGSliceTracker(start_time=5)) == events[2:]
    assert storage.to_events(DGSliceTracker(start_time=5, end_time=9)) == events[2:-2]
    assert storage.to_events(DGSliceTracker(node_slice={1, 2, 3})) == events[0:2] + [
        events[3]
    ] + [events[-1]]
    assert (
        storage.to_events(
            DGSliceTracker(start_time=6, end_time=9, node_slice={1, 2, 3})
        )
        == []
    )


def test_init_incompatible_node_feature_dimension(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=3, features=torch.rand(2, 5)),
        EdgeEvent(t=5, src=10, dst=20, features=torch.rand(3, 6)),
        NodeEvent(t=6, src=7, features=torch.rand(3, 6)),
    ]
    with pytest.raises(ValueError):
        _ = DGStorageImpl(events)


def test_init_incompatible_edge_feature_dimension(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=3, features=torch.rand(2, 5)),
        NodeEvent(t=5, src=10, features=torch.rand(2, 5)),
        NodeEvent(t=6, src=7, features=torch.rand(3, 6)),
    ]
    with pytest.raises(ValueError):
        _ = DGStorageImpl(events)


def test_init_out_of_order_events_list(DGStorageImpl, events_list_out_of_time_order):
    with pytest.warns(UserWarning):
        _ = DGStorageImpl(events_list_out_of_time_order)


def test_init_non_event_type(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=3),
        'foo',  # Should raise
    ]
    with pytest.raises(ValueError):
        _ = DGStorageImpl(events)


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_get_start_time_node_only_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_start_time(DGSliceTracker()) == events[0].t
    assert storage.get_start_time(DGSliceTracker(node_slice={4, 5})) == 5
    assert storage.get_start_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_get_start_time_edge_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_start_time(DGSliceTracker()) == events[0].t
    assert storage.get_start_time(DGSliceTracker(node_slice={4, 5})) == 5
    assert storage.get_start_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_get_start_time_events_list_with_multi_events_per_timestamp(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_start_time(DGSliceTracker()) == events[0].t
    assert storage.get_start_time(DGSliceTracker(node_slice={4, 5})) == 5
    assert storage.get_start_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_get_end_time_node_only_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_end_time(DGSliceTracker()) == events[-1].t
    assert storage.get_end_time(DGSliceTracker(node_slice={2, 3})) == 1
    assert storage.get_end_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_get_end_time_edge_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_end_time(DGSliceTracker()) == events[-1].t
    assert storage.get_end_time(DGSliceTracker(node_slice={2, 3})) == 5
    assert storage.get_end_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_get_end_time_events_list_with_multi_events_per_timestamp(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_end_time(DGSliceTracker()) == events[-1].t
    assert storage.get_end_time(DGSliceTracker(node_slice={2, 3})) == 5
    assert storage.get_end_time(DGSliceTracker(node_slice={100})) == None


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_get_nodes_node_only_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_nodes(DGSliceTracker()) == set([2, 4, 6])
    assert storage.get_nodes(DGSliceTracker(start_time=5)) == set([4, 6])
    assert storage.get_nodes(DGSliceTracker(end_time=4)) == set([2])
    assert storage.get_nodes(DGSliceTracker(start_time=5, end_time=9)) == set([4])


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_get_nodes_edge_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_nodes(DGSliceTracker()) == set([2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(start_time=5)) == set([2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(end_time=4)) == set([2])
    assert storage.get_nodes(DGSliceTracker(start_time=5, end_time=9)) == set([2, 4])


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_get_nodes_events_list_with_multi_events_per_timestamp(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_nodes(DGSliceTracker()) == set([1, 2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(start_time=5)) == set([1, 2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(end_time=4)) == set([2])
    assert storage.get_nodes(DGSliceTracker(start_time=5, end_time=9)) == set([2, 4])


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_get_edges_node_only_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    expected = torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
    torch.testing.assert_close(storage.get_edges(DGSliceTracker()), expected)
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5)), expected
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker(end_time=4)), expected)
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5, end_time=9)), expected
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(node_slice={1, 2, 3})), expected
    )
    torch.testing.assert_close(
        storage.get_edges(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        ),
        expected,
    )


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_get_edges_edge_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)

    expected = (
        torch.tensor([2, 2, 6], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker()), expected)

    expected = (
        torch.tensor([2, 6], dtype=torch.int64),
        torch.tensor([4, 8], dtype=torch.int64),
        torch.tensor([5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5)), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker(end_time=4)), expected)

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5, end_time=9)), expected
    )

    expected = (
        torch.tensor([2, 2], dtype=torch.int64),
        torch.tensor([2, 4], dtype=torch.int64),
        torch.tensor([1, 5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(node_slice={1, 2, 3})), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        ),
        expected,
    )


def test_get_edges_events_list_with_multi_events_per_timestamp(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=3),
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=4, dst=9),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=10, src=6, dst=10),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    storage = DGStorageImpl(events)

    expected = (
        torch.tensor([2, 2, 4, 2, 6, 1], dtype=torch.int64),
        torch.tensor([3, 2, 9, 4, 10, 8], dtype=torch.int64),
        torch.tensor([1, 1, 5, 5, 10, 20], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker()), expected)

    expected = (
        torch.tensor([4, 2, 6, 1], dtype=torch.int64),
        torch.tensor([9, 4, 10, 8], dtype=torch.int64),
        torch.tensor([5, 5, 10, 20], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5)), expected
    )

    expected = (
        torch.tensor([2, 2], dtype=torch.int64),
        torch.tensor([3, 2], dtype=torch.int64),
        torch.tensor([1, 1], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker(end_time=4)), expected)

    expected = (
        torch.tensor([4, 2], dtype=torch.int64),
        torch.tensor([9, 4], dtype=torch.int64),
        torch.tensor([5, 5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5, end_time=9)), expected
    )

    expected = (
        torch.tensor([2, 2, 2], dtype=torch.int64),
        torch.tensor([3, 2, 4], dtype=torch.int64),
        torch.tensor([1, 1, 5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(node_slice={2, 3})), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        ),
        expected,
    )


@pytest.mark.parametrize(
    'events', ['node_only_events_list', 'node_only_events_list_with_features']
)
def test_get_num_timestamps_node_only_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_num_timestamps(DGSliceTracker()) == 3
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5)) == 2
    assert storage.get_num_timestamps(DGSliceTracker(end_time=4)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5, end_time=9)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(node_slice={2, 3})) == 1
    assert (
        storage.get_num_timestamps(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        == 0
    )


@pytest.mark.parametrize(
    'events', ['edge_only_events_list', 'edge_only_events_list_with_features']
)
def test_get_num_timestamps_edge_events_list(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_num_timestamps(DGSliceTracker()) == 3
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5)) == 2
    assert storage.get_num_timestamps(DGSliceTracker(end_time=4)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5, end_time=9)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(node_slice={2, 3})) == 2
    assert (
        storage.get_num_timestamps(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        == 1
    )


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_get_num_timetamps_events_list_with_multi_events_per_timestamp(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_num_timestamps(DGSliceTracker()) == 4
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5)) == 3
    assert storage.get_num_timestamps(DGSliceTracker(end_time=4)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5, end_time=9)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(node_slice={2, 3})) == 2
    assert (
        storage.get_num_timestamps(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        == 1
    )


@pytest.mark.parametrize(
    'events',
    [
        'events_list_with_multi_events_per_timestamp',
        'events_list_with_features_multi_events_per_timestamp',
    ],
)
def test_get_num_events_list_with_multi_events_per_timestamp(
    DGStorageImpl, events, request
):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_num_events(DGSliceTracker()) == 6
    assert storage.get_num_events(DGSliceTracker(start_time=5)) == 4
    assert storage.get_num_events(DGSliceTracker(end_time=4)) == 2
    assert storage.get_num_events(DGSliceTracker(start_time=5, end_time=9)) == 2
    assert storage.get_num_events(DGSliceTracker(node_slice={2, 3})) == 3
    assert (
        storage.get_num_events(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        == 1
    )


@pytest.mark.parametrize(
    'events',
    [
        'node_only_events_list',
        'node_only_events_list_with_features',
        'edge_only_events_list',
        'events_list_with_multi_events_per_timestamp',
    ],
)
def test_get_edge_feats_no_edge_feats(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_edge_feats(DGSliceTracker()) is None
    assert storage.get_edge_feats(DGSliceTracker(start_time=5)) is None
    assert storage.get_edge_feats(DGSliceTracker(end_time=4)) is None
    assert storage.get_edge_feats(DGSliceTracker(start_time=5, end_time=9)) is None
    assert storage.get_edge_feats(DGSliceTracker(node_slice={2, 3})) is None
    assert (
        storage.get_edge_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


def test_get_edge_feats_edge_events_list(
    DGStorageImpl, edge_only_events_list_with_features
):
    events = edge_only_events_list_with_features
    storage = DGStorageImpl(events)

    exp_edge_feats = torch.zeros(11, 8 + 1, 8 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[0].features
    exp_edge_feats[5, 2, 4] = events[1].features
    exp_edge_feats[10, 6, 8] = events[2].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker()).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(11, 8 + 1, 8 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[1].features
    exp_edge_feats[10, 6, 8] = events[2].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(5, 2 + 1, 2 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[0].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(end_time=4)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(10, 4 + 1, 4 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_edge_feats,
    )
    exp_edge_feats = torch.zeros(6, 4 + 1, 4 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[0].features
    exp_edge_feats[5, 2, 4] = events[1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(node_slice={1, 2, 3})).to_dense(),
        exp_edge_feats,
    )

    exp_edge_feats = torch.zeros(10, 4 + 1, 4 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[1].features
    assert torch.equal(
        storage.get_edge_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        ).to_dense(),
        exp_edge_feats,
    )

    assert (
        storage.get_edge_feats(
            DGSliceTracker(start_time=6, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


def test_get_edge_feats_with_multi_events_per_timestamp(
    DGStorageImpl, events_list_with_features_multi_events_per_timestamp
):
    events = events_list_with_features_multi_events_per_timestamp
    storage = DGStorageImpl(events)

    exp_edge_feats = torch.zeros(21, 8 + 1, 8 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker()).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(21, 8 + 1, 8 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(5, 2 + 1, 2 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(end_time=4)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(10, 4 + 1, 4 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_edge_feats,
    )

    exp_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(node_slice={1, 2, 3})).to_dense(),
        exp_edge_feats,
    )

    exp_edge_feats = torch.zeros(10, 4 + 1, 4 + 1, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(
        storage.get_edge_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        ).to_dense(),
        exp_edge_feats,
    )

    assert (
        storage.get_edge_feats(
            DGSliceTracker(start_time=6, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


@pytest.mark.parametrize(
    'events',
    [
        'edge_only_events_list',
        'edge_only_events_list_with_features',
        'node_only_events_list',
        'events_list_with_multi_events_per_timestamp',
    ],
)
def test_get_node_feats_no_node_feats(DGStorageImpl, events, request):
    events = request.getfixturevalue(events)
    storage = DGStorageImpl(events)
    assert storage.get_node_feats(DGSliceTracker()) is None
    assert storage.get_node_feats(DGSliceTracker(start_time=5)) is None
    assert storage.get_node_feats(DGSliceTracker(end_time=4)) is None
    assert storage.get_node_feats(DGSliceTracker(start_time=5, end_time=9)) is None
    assert storage.get_node_feats(DGSliceTracker(node_slice={2, 3})) is None
    assert (
        storage.get_node_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


def test_get_node_feats_node_events_list(
    DGStorageImpl, node_only_events_list_with_features
):
    events = node_only_events_list_with_features
    storage = DGStorageImpl(events)

    exp_node_feats = torch.zeros(11, 6 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[1].features
    exp_node_feats[10, 6] = events[2].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker()).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(11, 6 + 1, 5)
    exp_node_feats[5, 4] = events[1].features
    exp_node_feats[10, 6] = events[2].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(start_time=5)).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(5, 2 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(end_time=4)).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(10, 4 + 1, 5)
    exp_node_feats[5, 4] = events[1].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(2, 2 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(node_slice={1, 2, 3})).to_dense(),
        exp_node_feats,
    )

    assert (
        storage.get_node_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


def test_get_node_feats_with_multi_events_per_timestamp(
    DGStorageImpl, events_list_with_features_multi_events_per_timestamp
):
    events = events_list_with_features_multi_events_per_timestamp
    storage = DGStorageImpl(events)

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker()).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(start_time=5)).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(5, 2 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(end_time=4)).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(10, 4 + 1, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[1, 2] = events[0].features
    assert torch.equal(
        storage.get_node_feats(DGSliceTracker(node_slice={1, 2, 3})).to_dense(),
        exp_node_feats,
    )

    assert (
        storage.get_node_feats(
            DGSliceTracker(start_time=5, end_time=9, node_slice={1, 2, 3})
        )
        is None
    )


@pytest.mark.skip('TODO: Add get_nbr')
def test_get_nbrs_single_hop(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    storage = DGStorageImpl(events)

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8], num_nbrs=[-1], slice=DGSliceTracker()
    )
    exp_nbrs = {
        1: [[(8, 20)]],
        2: [[(2, 1), (4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[(1, 20)]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(start_time=5),
    )
    exp_nbrs = {
        1: [[(8, 20)]],
        2: [[(4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[(1, 20)]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(start_time=5, end_time=9),
    )
    exp_nbrs = {
        1: [[]],
        2: [[(4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(node_slice={1, 2, 3}),
    )
    exp_nbrs = {
        1: [[]],
        2: [[(2, 1)]],
        3: [[]],
        4: [[]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(node_slice={2, 3}),
    )
    exp_nbrs = {
        1: [[]],
        2: [[(2, 1)]],
        3: [[]],
        4: [[]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(node_slice={2, 3}, end_time=4),
    )
    exp_nbrs = {
        1: [[]],
        2: [[(2, 1)]],
        3: [[]],
        4: [[]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])


@pytest.mark.skip('TODO: Add get_nbr tests')
def test_get_nbrs_single_hop_sampling_required(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    storage = DGStorageImpl(events)

    nbrs = storage.get_nbrs(seed_nodes=[2], num_nbrs=[1], slice=DGSliceTracker())
    exp_nbrs = {
        2: [[(2, 1)]],
    }
    # TODO: Either return a set or make this easier to check
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[2], num_nbrs=[1], slice=DGSliceTracker(end_time=4)
    )
    exp_nbrs = {
        2: [[(2, 1)]],
    }
    # TODO: Either return a set or make this easier to check
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])


@pytest.mark.skip('TODO: Add get_nbr tests')
def test_get_nbrs_single_hop_duplicate_edges_at_different_time(DGStorageImpl):
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
        EdgeEvent(t=100, src=2, dst=2),
        EdgeEvent(t=500, src=2, dst=4),
        EdgeEvent(t=2000, src=1, dst=8),
    ]
    storage = DGStorageImpl(events)

    nbrs = storage.get_nbrs(seed_nodes=[2], num_nbrs=[-1], slice=DGSliceTracker())
    exp_nbrs = {
        2: [[(2, 1), (4, 5), (2, 100), (4, 500)]],
    }
    # TODO: Either return a set or make this easier to check
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])


@pytest.mark.skip(reason='Multi-hop get_nbrs not implemented')
def test_get_nbrs_multiple_hops(DGStorageImpl):
    pass


@pytest.mark.skip(reason='TODO: Add test with event idx constraints!')
def test_dg_storage_with_event_contraints(DGStorageImpl):
    pass


def test_get_dg_storage_backend():
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_class():
    set_dg_storage_backend(DGStorageArrayBackend)
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_str():
    set_dg_storage_backend('ArrayBackend')
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_bad_str():
    with pytest.raises(ValueError):
        set_dg_storage_backend('foo')
