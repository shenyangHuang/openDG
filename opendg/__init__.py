""" OpenDG library for geometric learning on temporal graphs."""

from .data.data import CTDG,DTDG
from .data.storage import EventStore

__version__ = '0.1.0'

__all__ = [
    'CTDG',
    'DTDG',
    'EventStore',
    '__version__',
]
