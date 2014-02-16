"""Compatibility routines."""

from __future__ import absolute_import

import sys

try:
    import Image  # pylint: disable=import-error,unused-import
except ImportError:
    from PIL import Image  # pylint: disable=unused-import

try:
    import ImageOps  # pylint: disable=import-error,unused-import
except ImportError:
    from PIL import ImageOps  # pylint: disable=unused-import

if sys.version_info[0] > 2:
    basestring = (str, bytes)  # pylint: disable=redefined-builtin,invalid-name
else:
    basestring = basestring  # pylint: disable=invalid-name
