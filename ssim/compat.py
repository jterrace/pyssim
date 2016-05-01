"""Compatibility routines."""

from __future__ import absolute_import

import sys

# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=redefined-variable-type
# pylint: disable=unused-import

try:
    import Image
except ImportError:
    from PIL import Image

try:
    import ImageOps
except ImportError:
    from PIL import ImageOps

if sys.version_info[0] > 2:
    basestring = (str, bytes)
else:
    basestring = basestring
