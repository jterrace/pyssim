"""Setup script for pyssim."""

from setuptools import find_packages
from setuptools import setup

install_requires = []  # pylint: disable=invalid-name

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
try:
    import PIL
except ImportError:
    try:
        import Image
    except ImportError:
        install_requires.append('pillow')

try:
    import numpy
except ImportError:
    install_requires.append('numpy')

try:
    import scipy
except ImportError:
    install_requires.append('scipy')

try:
    import argparse
except ImportError:
    install_requires.append('argparse')

setup(
    name='pyssim',
    version='0.3',
    description=('Module for computing Structured Similarity Image Metric '
                 '(SSIM) in Python'),
    author='Antoine Vacavant, Christopher Godfrey, Jeff Terrace',
    author_email='jterrace@gmail.com',
    platforms=['any'],
    license='MIT License',
    install_requires=install_requires,
    url='https://github.com/jterrace/pyssim',
    entry_points={
        'console_scripts': [
            'pyssim = ssim.__main__:main'
        ]
    },
    packages=find_packages()
)
