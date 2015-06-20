"""Contains SSIM library functions and classes."""

from __future__ import absolute_import

import argparse
import glob
import sys

import numpy as np
from scipy import signal

from ssim import compat
from ssim.compat import Image, ImageOps
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
    def __init__(self, img, gaussian_kernel_1d=None, size=None):
        """Create an SSIMImage.

        Args:
          img (str or PIL.Image): PIL Image object or file name.
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): New image size to resize image to.
        """
        # Use existing or create a new PIL.Image
        self.img = img if not isinstance(img, compat.basestring) \
            else compat.Image.open(img)

        # Resize image if size is defined and different
        # from original image
        if size and size != self.img.size:
            self.img = self.img.resize(size, Image.ANTIALIAS)

        # Set the size of the image
        self.size = self.img.size

        # If gaussian kernel is defined we create
        # common SSIM objects
        if gaussian_kernel_1d is not None:

            self.gaussian_kernel_1d = gaussian_kernel_1d

            # np.array of grayscale and alpha image
            self.img_gray, self.img_alpha = to_grayscale(self.img)
            if self.img_alpha is not None:
                self.img_gray[self.img_alpha == 255] = 0

            # Squared grayscale
            self.img_gray_squared = self.img_gray ** 2

            # Convolve grayscale image with gaussian
            self.img_gray_mu = convolve_gaussian_2d(
                self.img_gray, self.gaussian_kernel_1d)

            # Squared mu
            self.img_gray_mu_squared = self.img_gray_mu ** 2

            # Convolve squared grayscale with gaussian
            self.img_gray_sigma_squared = convolve_gaussian_2d(
                self.img_gray_squared, self.gaussian_kernel_1d)

            # Substract squared mu
            self.img_gray_sigma_squared -= self.img_gray_mu_squared

        # If we don't define gaussian kernel, we create
        # common CW-SSIM objects
        else:
            # Grayscale PIL.Image
            self.img_gray = ImageOps.grayscale(self.img)

class SSIM(object):
    """Computes SSIM between two images."""
    def __init__(self, img, gaussian_kernel_1d=None, size=None,
                 l=255, k_1=0.01, k_2=0.03, k=0.01):
        """Create an SSIM object.

        Args:
          img (str or PIL.Image): Reference image to compare other images to.
          l, k_1, k_2 (float): SSIM configuration variables.
          k (float): CW-SSIM configuration variable (default 0.01)
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): resize the image to the tuple size
        """
        self.k = k
        # Set k1,k2 & c1,c2 to depend on L (width of color map).
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = SSIMImage(img, gaussian_kernel_1d, size)

    def ssim_value(self, target):
        """Compute the SSIM value from the reference image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).

        Returns:
          Computed SSIM float value.
        """
        # Performance boost if handed a compatible SSIMImage object.
        if not isinstance(target, SSIMImage) \
          or not np.array_equal(self.gaussian_kernel_1d,
                                target.gaussian_kernel_1d):
            target = SSIMImage(target, self.gaussian_kernel_1d, self.img.size)

        img_mat_12 = self.img.img_gray * target.img_gray
        img_mat_sigma_12 = convolve_gaussian_2d(
            img_mat_12, self.gaussian_kernel_1d)
        img_mat_mu_12 = self.img.img_gray_mu * target.img_gray_mu
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

        # Numerator of SSIM
        num_ssim = ((2 * img_mat_mu_12 + self.c_1) *
                    (2 * img_mat_sigma_12 + self.c_2))

        # Denominator of SSIM
        den_ssim = (
            (self.img.img_gray_mu_squared + target.img_gray_mu_squared +
             self.c_1) *
            (self.img.img_gray_sigma_squared +
             target.img_gray_sigma_squared + self.c_2))

        ssim_map = num_ssim / den_ssim
        index = np.average(ssim_map)
        return index

    def cw_ssim_value(self, target, width=30):
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).
          width: width for the wavelet convolution (default: 30)

        Returns:
          Computed CW-SSIM float value.
        """
        if not isinstance(target, SSIMImage):
            target = SSIMImage(target, size=self.img.size)

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)

        # Use the image data as arrays
        sig1 = np.asarray(self.img.img_gray.getdata())
        sig2 = np.asarray(target.img_gray.getdata())

        # Convolution
        cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
        cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)

        # Compute the first term
        c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
        c1_2 = np.square(abs(cwtmatr1))
        c2_2 = np.square(abs(cwtmatr2))
        num_ssim_1 = 2 * np.sum(c1c2, axis=0) + self.k
        den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + self.k

        # Compute the second term
        c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
        num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + self.k
        den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + self.k

        # Construct the result
        ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)

        # Average the per pixel results
        index = np.average(ssim_map)
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
    parser.add_argument('--cw', help='compute the complex wavelet SSIM',
                        action='store_true')
    parser.add_argument(
        'base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument(
        'comparison_images', metavar='image path with* or image2.png')
    parser.add_argument('--width', type=int, default=None,
                        help='scales the image before computing SSIM')
    parser.add_argument('--height', type=int, default=None,
                        help='scales the image before computing SSIM')

    args = parser.parse_args()

    if args.width and args.height:
        size = (args.width, args.height)
    else:
        size = None

    if not args.cw:
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel_1d = get_gaussian_kernel(
            gaussian_kernel_width, gaussian_kernel_sigma)

    comparison_images = glob.glob(args.comparison_images)
    is_a_single_image = len(comparison_images) == 1

    for comparison_image in comparison_images:

        if args.cw:
            ssim = SSIM(args.base_image.name, size=size)
            ssim_value = ssim.cw_ssim_value(comparison_image)
        else:
            ssim = SSIM(args.base_image.name, gaussian_kernel_1d, size=size)
            ssim_value = ssim.ssim_value(comparison_image)

        if is_a_single_image:
            sys.stdout.write('%.7g' % ssim_value)
        else:
            sys.stdout.write('%s - %s: %.7g' % (
                args.base_image.name, comparison_image, ssim_value))
        sys.stdout.write('\n')

if __name__ == '__main__':
    main()
