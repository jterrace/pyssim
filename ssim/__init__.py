from ssim import get_gaussian_kernel, SSIM, SSIMImage
from compat import Image

def compute_ssim(image1, image2, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11):
    """
    The function to compute SSIM
    @param im1: PIL Image object
    @param im2: PIL Image object
    @return: SSIM float value
    """
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    return SSIM(image1, gaussian_kernel_1d).ssim_value(image2)
