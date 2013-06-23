"""
This module computes the Structured Similarity Image Metric (SSIM)

Created on 21 nov. 2011
@author: Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr, http://isit.u-clermont1.fr/~anvacava

Modified by Christopher Godfrey, on 17 July 2012 (lines 32-34)
Modified by Jeff Terrace, starting 29 August 2012
"""

import numpy
import scipy.ndimage
from numpy.ma.core import exp, sqrt
from scipy.constants.constants import pi
from compat import ImageOps

def _to_grayscale(im):
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

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    result = scipy.ndimage.filters.correlate1d(image, gaussian_kernel_1d, axis = 0)
    result = scipy.ndimage.filters.correlate1d(result, gaussian_kernel_1d, axis = 1)
    return result

def compute_ssim(im1, im2, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11):
    """
    The function to compute SSIM
    @param im1: PIL Image object
    @param im2: PIL Image object
    @return: SSIM float value
    """
    
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = numpy.ndarray((gaussian_kernel_width))
    mu = int(gaussian_kernel_width / 2)

    #Fill Gaussian kernel
    for i in xrange(gaussian_kernel_width):
            gaussian_kernel_1d[i] = (1 / (sqrt(2 * pi) * (gaussian_kernel_sigma))) * \
                exp(-(((i - mu) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # convert the images to grayscale
    img_mat_1, img_alpha_1 = _to_grayscale(im1)
    img_mat_2, img_alpha_2 = _to_grayscale(im2)
    
    # don't count pixels where both images are both fully transparent
    if img_alpha_1 is not None and img_alpha_2 is not None:
        img_mat_1[img_alpha_1 == 255] = 0
        img_mat_2[img_alpha_2 == 255] = 0
    
    #Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = convolve_gaussian_2d(img_mat_1, gaussian_kernel_1d)
    img_mat_mu_2 = convolve_gaussian_2d(img_mat_2, gaussian_kernel_1d)
    
    #Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = convolve_gaussian_2d(img_mat_1_sq, gaussian_kernel_1d)
    img_mat_sigma_2_sq = convolve_gaussian_2d(img_mat_2_sq, gaussian_kernel_1d)
    
    #Covariance
    img_mat_sigma_12 = convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)
    
    #Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12
    
    #set k1,k2 & c1,c2 to depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2
    
    #Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    
    #Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    
    #SSIM
    ssim_map = num_ssim / den_ssim
    index = numpy.average(ssim_map)

    return index
