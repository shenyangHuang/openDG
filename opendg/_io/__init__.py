from opendg._io.csv import read_csv
from opendg._io.pandas import read_pandas
from opendg._io.tgb import read_tgb

from typing import Any, List
import pathlib

import pandas as pd

from opendg.events import Event


def read_events(data: str | pathlib.Path | pd.DataFrame, **kwargs: Any) -> List[Event]:
    if isinstance(data, pd.DataFrame):
        return read_pandas(data, **kwargs)
    elif isinstance(data, (str, pathlib.Path)):
        return read_csv(data, **kwargs)
    elif isinstance(data, str) and str(data).startswith('tgb'):
        return read_tgb(name=str(data), **kwargs)
    else:
        raise ValueError('cannot read events from type {type(data)}')
