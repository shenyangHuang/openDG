import csv
import tempfile

import torch

from opendg._io import read_csv
from opendg.events import EdgeEvent


def test_csv_conversion_no_features():
    events = [
        EdgeEvent(t=1, src=2, dst=3),
        EdgeEvent(t=1, src=10, dst=20),
    ]

    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(events, f.name, **col_names)
        recovered_events = read_csv(f.name, **col_names)
    assert events == recovered_events


def test_csv_conversion_with_features():
    events = [
        EdgeEvent(t=1, src=2, dst=3, features=torch.rand(5)),
        EdgeEvent(t=5, src=10, dst=20, features=torch.rand(5)),
    ]

    edge_feature_col = [f'dim_{i}' for i in range(5)]
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(events, f.name, edge_feature_col=edge_feature_col, **col_names)
        recovered_events = read_csv(
            f.name, edge_feature_col=edge_feature_col, **col_names
        )

    assert len(events) == len(recovered_events)
    for i in range(len(recovered_events)):
        assert isinstance(events[i], EdgeEvent)
        assert isinstance(recovered_events[i], EdgeEvent)
        assert events[i].t == recovered_events[i].t
        assert events[i].edge == recovered_events[i].edge
        torch.testing.assert_close(events[i].features, recovered_events[i].features)


def _write_csv(events, fp, src_col, dst_col, time_col, edge_feature_col=None):
    with open(fp, 'w', newline='') as f:
        fieldnames = [src_col, dst_col, time_col]
        if edge_feature_col is not None:
            fieldnames += edge_feature_col
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            assert isinstance(event, EdgeEvent)
            row = {src_col: event.src, dst_col: event.dst, time_col: event.t}
            if event.features is not None:
                if edge_feature_col is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                if len(event.features.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(event.features) != len(edge_feature_col):
                    raise ValueError(
                        f'Got {len(event.features)}-dimensional feature tensor but only '
                        f'specified {len(edge_feature_col)} feature column names.'
                    )

                features_list = event.features.tolist()
                for feature_col, feature_val in zip(edge_feature_col, features_list):
                    row[feature_col] = feature_val

            writer.writerow(row)
