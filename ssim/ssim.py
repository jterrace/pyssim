import argparse
from compat import Image
import numpy
import scipy.ndimage
from numpy.ma.core import exp, sqrt
from scipy.constants.constants import pi
from compat import ImageOps
from utils import to_grayscale, convolve_gaussian_2d, get_gaussian_kernel

class SSIM(object):
    def __init__(self, img1, img2, gaussian_kernel_1d, l =255, k_1 =0.01, k_2 = 0.03):
        #set k1,k2 & c1,c2 to depend on L (width of color map)
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.img1 = img1
        self.img2 = img2
        self.img_mat_12 = self.img1.img_gray * self.img2.img_gray
        self.img_mat_sigma_12 = convolve_gaussian_2d(self.img_mat_12, gaussian_kernel_1d)
        self.img_mat_mu_12 = self.img1.img_gray_mu * self.img2.img_gray_mu
        self.img_mat_sigma_12 = self.img_mat_sigma_12 - self.img_mat_mu_12

    def ssim_value(self):
        #Numerator of SSIM
        num_ssim = (2 * self.img_mat_mu_12 + self.c_1) * (2 * self.img_mat_sigma_12 + self.c_2)
        #Denominator of SSIM
        den_ssim = (self.img1.img_gray_mu_squared + self.img2.img_gray_mu_squared + self.c_1) * \
                   (self.img1.img_gray_sigma_squared + self.img2.img_gray_sigma_squared + self.c_2)
        #SSIM
        ssim_map = num_ssim / den_ssim
        index = numpy.average(ssim_map)
        return index

class SSIMImage(object):
    def __init__(self, img, gaussian_kernel_1d):
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = img
        self.img_gray, self.img_alpha = to_grayscale(img)
        if self.img_alpha is not None:
            self.img_gray[self.img_alpha == 255] = 0

        self.img_gray_squared = self.img_gray ** 2
        self.img_gray_mu = convolve_gaussian_2d(self.img_gray,  self.gaussian_kernel_1d)
        self.img_gray_mu_squared = self.img_gray_mu ** 2
        self.img_gray_sigma_squared = convolve_gaussian_2d(self.img_gray_squared, self.gaussian_kernel_1d)
        self.img_gray_sigma_squared -= self.img_gray_mu_squared

def compute_ssim(im1, im2, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11):
    """
    The function to compute SSIM
    @param im1: PIL Image object
    @param im2: PIL Image object
    @return: SSIM float value
    """
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    img1 = SSIMImage(im1, gaussian_kernel_1d)
    img2 = SSIMImage(im2, gaussian_kernel_1d)
    return SSIM(img1, img2, gaussian_kernel_1d).ssim_value()

def main():
    parser = argparse.ArgumentParser(prog="pyssim",
                                     description="Compares two images using the SSIM metric")
    parser.add_argument('base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument('comparison_image', metavar='image2.png', type=argparse.FileType('r'))
    args = parser.parse_args()
    
    im1 = Image.open(args.base_image)
    im2 = Image.open(args.comparison_image)
    
    print compute_ssim(im1, im2)
    
if __name__ == '__main__':
    main()
