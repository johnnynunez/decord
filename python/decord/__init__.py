"""Decord python package"""
from . import function
from .version import __version__

from ._ffi.runtime_ctypes import TypeCode
from ._ffi.function import register_func, get_global_func, list_global_func_names, extract_ext_funcs
from ._ffi.base import DECORDError, DECORDLimitReachedError

from .base import ALL

from . import ndarray as nd
from .ndarray import cpu, gpu
from . import bridge
from . import logging
from .video_reader import VideoReader
from .video_loader import VideoLoader
from .audio_reader import AudioReader
from .av_reader import AVReader

logging.set_level(logging.ERROR)