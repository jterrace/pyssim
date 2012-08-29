# pyssim

This module implements the Structural Similarity Image Metric (SSIM). Original code written by Antoine Vacavant from http://isit.u-clermont1.fr/~anvacava/code.html, with modifications by Christopher Godfrey and Jeff Terrace.

## Installation

    pip install pyssim

## Running

    $ pyssim --help
    usage: pyssim [-h] image1.png image2.png

    Compares two images using the SSIM metric

    positional arguments:
      image1.png
      image2.png

    optional arguments:
      -h, --help  show this help message and exit

## References

* [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600--612, 2004. 
* [2] Z. Wang and A. C. Bovik. Mean squared error: Love it or leave it? - A new look at signal fidelity measures. IEEE Signal Processing Magazine, 26(1):98--117, 2009.