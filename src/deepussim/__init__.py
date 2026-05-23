"""deepussim — Genesis-based ultrasound simulation from CBCT.

The geometric core (geometry, us.reslice, us.renderer, calib.registration) depends
only on numpy/scipy. The ``sim`` subpackage imports Genesis lazily, so importing
``deepussim`` never requires Genesis to be installed.
"""

__version__ = "0.0.1"

from . import geometry  # noqa: F401

__all__ = ["geometry", "__version__"]
