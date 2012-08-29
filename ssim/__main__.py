import argparse
import ssim
from ssim.compat import Image

def main():
    parser = argparse.ArgumentParser(prog="pyssim",
                                     description="Compares two images using the SSIM metric")
    parser.add_argument('base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument('comparison_image', metavar='image2.png', type=argparse.FileType('r'))
    args = parser.parse_args()
    
    im1 = Image.open(args.base_image)
    im2 = Image.open(args.comparison_image)
    
    print ssim.compute_ssim(im1, im2)
    
if __name__ == '__main__':
    main()
