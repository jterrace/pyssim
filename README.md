# pyssim

This module implements the Structural Similarity Image Metric (SSIM).
Original code written by Antoine Vacavant from
http://isit.u-clermont1.fr/~anvacava/code.html, with modifications by
Christopher Godfrey and Jeff Terrace.

[![Build Status](https://secure.travis-ci.org/jterrace/pyssim.png)](http://travis-ci.org/#!/jterrace/pyssim)

## Installation

    pip install pyssim

## Running

    $ pyssim --help
    usage: pyssim [-h] image1.png image path with* or image2.png

    Compares an image with a list of images using the SSIM metric.
      Example:
        pyssim test-images/test1-1.png "test-images/*"

    positional arguments:
      image1.png
      image path with* or image2.png

    optional arguments:
      -h, --help            show this help message and exit
      --cw                  compute the complex wavelet SSIM
      --width WIDTH         scales the image before computing SSIM
      --height HEIGHT       scales the image before computing SSIM

## Compatibility

pyssim is known to work with Python 2.7 and 3.2 and we test these versions on
Travis CI to make sure they keep working. 2.6 and 3.3 will probably work, but
we omit them from testing due to complications with setting them up on Travis
CI.

## Development

To run from a local git client:

    PYTHONPATH="." python ssim

To run the lint checks:

    pylint --rcfile=.pylintrc -r n ssim setup.py

To test:

    $ PYTHONPATH="." python ssim test-images/test1-1.png "test-images/*"
    test-images/test1-1.png - test-images/test1-1.png: 1
    test-images/test1-1.png - test-images/test1-2.png: 0.9980119
    test-images/test1-1.png - test-images/test2-1.png: 0.6726952
    test-images/test1-1.png - test-images/test2-2.png: 0.6485879

## References

* [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600--612, 2004. 
* [2] Z. Wang and A. C. Bovik. Mean squared error: Love it or leave it? - A new look at signal fidelity measures. IEEE Signal Processing Magazine, 26(1):98--117, 2009.
* [3] Z. Wang and E.P. Simoncelli. Translation Insensitive Image Similarity in Complex Wavelet Domain. Acoustics, Speech, and Signal Processing, 2005. Proceedings. (ICASSP '05). IEEE International Conference on , vol.2, no., pp.573,576, March 18-23, 2005