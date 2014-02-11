import argparse
import glob
from compat import Image
import numpy
import scipy.ndimage
from numpy.ma.core import exp, sqrt
from scipy.constants.constants import pi
from utils import to_grayscale, convolve_gaussian_2d, get_gaussian_kernel

class SSIMImage(object):
    def __init__(self, img, gaussian_kernel_1d, size = None):
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = Image.open(img)
        if size:
            self.img = self.img.resize(size)
        self.size = self.img.size
        self.img_gray, self.img_alpha = to_grayscale(self.img)
        if self.img_alpha is not None:
            self.img_gray[self.img_alpha == 255] = 0
        self.img_gray_squared = self.img_gray ** 2
        self.img_gray_mu = convolve_gaussian_2d(self.img_gray,  self.gaussian_kernel_1d)
        self.img_gray_mu_squared = self.img_gray_mu ** 2
        self.img_gray_sigma_squared = convolve_gaussian_2d(self.img_gray_squared, self.gaussian_kernel_1d)
        self.img_gray_sigma_squared -= self.img_gray_mu_squared

class SSIM(object):
    def __init__(self, img1, gaussian_kernel_1d, l =255, k_1 =0.01, k_2 = 0.03):
        #set k1,k2 & c1,c2 to depend on L (width of color map)
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img1 = SSIMImage(img1, gaussian_kernel_1d)

    def ssim_value(self, img2):
        self.img2 = SSIMImage(img2, self.gaussian_kernel_1d,self.img1.size)
        self.img_mat_12 = self.img1.img_gray * self.img2.img_gray
        self.img_mat_sigma_12 = convolve_gaussian_2d(self.img_mat_12, self.gaussian_kernel_1d)
        self.img_mat_mu_12 = self.img1.img_gray_mu * self.img2.img_gray_mu
        self.img_mat_sigma_12 = self.img_mat_sigma_12 - self.img_mat_mu_12
        #Numerator of SSIM
        num_ssim = (2 * self.img_mat_mu_12 + self.c_1) * (2 * self.img_mat_sigma_12 + self.c_2)
        #Denominator of SSIM
        den_ssim = (self.img1.img_gray_mu_squared + self.img2.img_gray_mu_squared + self.c_1) * \
                   (self.img1.img_gray_sigma_squared + self.img2.img_gray_sigma_squared + self.c_2)
        #SSIM
        ssim_map = num_ssim / den_ssim
        index = numpy.average(ssim_map)
        return index

def main():
    parser = argparse.ArgumentParser(prog="pyssim",
                                     description="Compares an image with a list of images using the SSIM metric")
    parser.add_argument('base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument('comparison_images', metavar='image path with* / image2')
    args = parser.parse_args()
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    
    comparison_images = glob.glob(args.comparison_images)
    is_a_single_image = len(comparison_images) == 1

    for comparison_image in comparison_images:
        try:
            ssim_value = SSIM(args.base_image.name, gaussian_kernel_1d).ssim_value(comparison_image)
            if is_a_single_image:
                print ssim_value
            else:
                print "%s - %s: %s" % (args.base_image.name, comparison_image, ssim_value)

        except Exception, e:
            print e
if __name__ == '__main__':
    main()
