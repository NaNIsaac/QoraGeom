"""Moved to ``qorageom.adapters.spec``; this module forwards to it for compatibility."""

import warnings

from qorageom.adapters import spec as _moved

warnings.warn(
    "qorageom.spec moved to qorageom.adapters.spec",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_moved, name)
