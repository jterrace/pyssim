"""Contains SSIM library functions and classes."""

from __future__ import absolute_import

import argparse
import glob
import sys

import numpy

from ssim import compat
from ssim.utils import convolve_gaussian_2d
from ssim.utils import get_gaussian_kernel
from ssim.utils import to_grayscale


class SSIMImage(object):
    """Wraps a PIL Image object with SSIM state.

    Attributes:
      img: Original PIL Image.
      img_gray: grayscale Image.
      img_gray_squared: squared img_gray.
      img_gray_mu: img_gray convolved with gaussian kernel.
      img_gray_mu_squared: squared img_gray_mu.
      img_gray_sigma_squared: img_gray convolved with gaussian kernel -
                              img_gray_mu_squared.
    """
    def __init__(self, img, gaussian_kernel_1d, size=None):
        """Create an SSIMImage.

        Args:
          img: PIL Image object or file name.
          gaussian_kernel_1d: Gaussian kernel from get_gaussian_kernel.
          size: New image size to resize image to.
        """
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = img
        if isinstance(img, compat.basestring):
            self.img = compat.Image.open(img)
        if size and size != self.img.size:
            self.img = self.img.resize(size)
        self.size = self.img.size
        self.img_gray, self.img_alpha = to_grayscale(self.img)
        if self.img_alpha is not None:
            self.img_gray[self.img_alpha == 255] = 0
        self.img_gray_squared = self.img_gray ** 2
        self.img_gray_mu = convolve_gaussian_2d(
            self.img_gray, self.gaussian_kernel_1d)
        self.img_gray_mu_squared = self.img_gray_mu ** 2
        self.img_gray_sigma_squared = convolve_gaussian_2d(
            self.img_gray_squared, self.gaussian_kernel_1d)
        self.img_gray_sigma_squared -= self.img_gray_mu_squared


class SSIM(object):
    """Computes SSIM between two images."""
    def __init__(self, img1, gaussian_kernel_1d, l=255, k_1=0.01, k_2=0.03):
        """Create an SSIM object.

        Args:
          img1: Reference image to compare other images to.
          gaussian_kernel_1d: Gaussian kernel from get_gaussian_kernel.
          l, k_1, k_2: SSIM configuration variables.
        """
        # Set k1,k2 & c1,c2 to depend on L (width of color map).
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img1 = SSIMImage(img1, gaussian_kernel_1d)

    def ssim_value(self, img2):
        """Compute the SSIM value from the reference image to the given image.

        Args:
          img2: Input image to compare the reference image to.

        Returns:
          Computed SSIM float value.
        """
        img2 = SSIMImage(img2, self.gaussian_kernel_1d, self.img1.size)
        img_mat_12 = self.img1.img_gray * img2.img_gray
        img_mat_sigma_12 = convolve_gaussian_2d(
            img_mat_12, self.gaussian_kernel_1d)
        img_mat_mu_12 = self.img1.img_gray_mu * img2.img_gray_mu
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

        # Numerator of SSIM
        num_ssim = ((2 * img_mat_mu_12 + self.c_1) *
                    (2 * img_mat_sigma_12 + self.c_2))

        # Denominator of SSIM
        den_ssim = (
            (self.img1.img_gray_mu_squared + img2.img_gray_mu_squared +
             self.c_1) *
            (self.img1.img_gray_sigma_squared +
             img2.img_gray_sigma_squared + self.c_2))

        ssim_map = num_ssim / den_ssim
        index = numpy.average(ssim_map)
        return index

def main():
    """Main function for pyssim."""
    description = '\n'.join([
        'Compares an image with a list of images using the SSIM metric.',
        '  Example:',
        '    pyssim test-images/test1-1.png "test-images/*"'
    ])

    parser = argparse.ArgumentParser(
        prog='pyssim', formatter_class=argparse.RawTextHelpFormatter,
        description=description)
    parser.add_argument(
        'base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument(
        'comparison_images', metavar='image path with* or image2.png')
    args = parser.parse_args()
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(
        gaussian_kernel_width, gaussian_kernel_sigma)

    comparison_images = glob.glob(args.comparison_images)
    is_a_single_image = len(comparison_images) == 1

    for comparison_image in comparison_images:
        ssim_value = SSIM(args.base_image.name, gaussian_kernel_1d).ssim_value(
            comparison_image)
        if is_a_single_image:
            sys.stdout.write('%.7g' % ssim_value)
        else:
            sys.stdout.write('%s - %s: %.7g' % (
                args.base_image.name, comparison_image, ssim_value))
        sys.stdout.write('\n')

if __name__ == '__main__':
    main()
