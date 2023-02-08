"""Common utility functions."""

from __future__ import absolute_import

import numpy as np
import scipy.ndimage

from ssim.compat import ImageOps


def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = scipy.ndimage.correlate1d(
        image, gaussian_kernel_1d, axis=0)
    return scipy.ndimage.correlate1d(
        result, gaussian_kernel_1d, axis=1)


def get_gaussian_kernel(gaussian_kernel_width=11, gaussian_kernel_sigma=1.5):
    """Generate a gaussian kernel."""
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = np.arange(0, gaussian_kernel_width, 1.)
    gaussian_kernel_1d -= gaussian_kernel_width / 2
    gaussian_kernel_1d = np.exp(-0.5 * gaussian_kernel_1d**2 /
                                gaussian_kernel_sigma**2)
    return gaussian_kernel_1d / np.sum(gaussian_kernel_1d)


def to_grayscale(img):
    """Convert PIL image to numpy grayscale array and numpy alpha array.

    Args:
      img (PIL.Image): PIL Image object.

    Returns:
      (gray, alpha): both numpy arrays.
    """
    gray = np.asarray(ImageOps.grayscale(img)).astype(float)

    imbands = img.getbands()
    alpha = None
    if 'A' in imbands:
        alpha = np.asarray(img.split()[-1]).astype(float)

    return gray, alpha
