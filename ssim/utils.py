"""Common utility functions."""

from __future__ import absolute_import

import numpy
from numpy.ma.core import exp
from numpy.ma.core import sqrt
import scipy.ndimage
from scipy.constants.constants import pi

from ssim.compat import ImageOps

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.filters.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    result = scipy.ndimage.filters.correlate1d(
        result, gaussian_kernel_1d, axis=1)
    return result

def get_gaussian_kernel(gaussian_kernel_width=11, gaussian_kernel_sigma=1.5):
    """Generate a gaussian kernel."""
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = numpy.ndarray((gaussian_kernel_width))
    norm_mu = int(gaussian_kernel_width / 2)

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        gaussian_kernel_1d[i] = (
            (1 / (sqrt(2 * pi) * (gaussian_kernel_sigma))) *
            exp(-(((i - norm_mu) ** 2)) / (2 * (gaussian_kernel_sigma ** 2))))
    return gaussian_kernel_1d

def to_grayscale(img):
    """Convert PIL image to numpy grayscale array and numpy alpha array.

    Args:
      img (PIL.Image): PIL Image object.

    Returns:
      (gray, alpha): both numpy arrays.
    """
    gray = numpy.asarray(ImageOps.grayscale(img)).astype(numpy.float)

    imbands = img.getbands()
    alpha = None
    if 'A' in imbands:
        alpha = numpy.asarray(img.split()[-1]).astype(numpy.float)

    return gray, alpha
