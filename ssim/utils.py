from compat import Image
import numpy
import scipy.ndimage
from numpy.ma.core import exp, sqrt
from scipy.constants.constants import pi
from compat import ImageOps

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    result = scipy.ndimage.filters.correlate1d(image, gaussian_kernel_1d, axis = 0)
    result = scipy.ndimage.filters.correlate1d(result, gaussian_kernel_1d, axis = 1)
    return result

def get_gaussian_kernel(gaussian_kernel_width=11, gaussian_kernel_sigma=1.5 ):
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = numpy.ndarray((gaussian_kernel_width))
    mu = int(gaussian_kernel_width / 2)

    #Fill Gaussian kernel
    for i in xrange(gaussian_kernel_width):
            gaussian_kernel_1d[i] = (1 / (sqrt(2 * pi) * (gaussian_kernel_sigma))) * \
                exp(-(((i - mu) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))
    return gaussian_kernel_1d

def to_grayscale(im):
    """
    Convert PIL image to numpy grayscale array and numpy alpha array
    @param im: PIL Image object
    @return (gray, alpha), both numpy arrays
    """
    gray = numpy.asarray(ImageOps.grayscale(im)).astype(numpy.float)
    
    imbands = im.getbands()
    if 'A' in imbands:
        alpha = numpy.asarray(im.split()[-1]).astype(numpy.float)
    else:
        alpha = None
    
    return gray, alpha
