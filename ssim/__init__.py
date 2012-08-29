"""
This module computes the Structured Similarity Image Metric (SSIM)

Created on 21 nov. 2011
@author: Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr, http://isit.u-clermont1.fr/~anvacava

Modified by Christopher Godfrey, on 17 July 2012 (lines 32-34)
Modified by Jeff Terrace, starting 29 August 2012
"""

import numpy
import scipy.ndimage
from numpy.ma.core import exp
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

def compute_ssim(im1, im2, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11):
    """
    The function to compute SSIM
    @param im1: PIL Image object
    @param im2: PIL Image object
    @return: SSIM float value
    """
    
    #Gaussian kernel definition
    gaussian_kernel = numpy.zeros((gaussian_kernel_width, gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

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
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)
    
    #Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)
    
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
