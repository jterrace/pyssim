"""Compatibility routines."""

from __future__ import absolute_import

try:
    import Image  # pylint: disable=import-error,unused-import
except ImportError:
    from PIL import Image  # pylint: disable=unused-import

try:
    import ImageOps  # pylint: disable=import-error,unused-import
except ImportError:
    from PIL import ImageOps  # pylint: disable=unused-import
