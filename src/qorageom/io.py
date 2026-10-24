"""Moved to ``qorageom.adapters.frames``; this module forwards to it for compatibility."""

import warnings

from qorageom.adapters import frames as _moved

warnings.warn(
    "qorageom.io moved to qorageom.adapters.frames",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)
