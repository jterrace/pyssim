from setuptools import find_packages, setup

install_requires = []

try: import matplotlib
except ImportError: install_requires.append('matplotlib')

try: import PIL
except ImportError:
    try: import Image
    except ImportError:
        install_requires.append('pillow')

try: import numpy
except ImportError: install_requires.append('numpy')

try: import scipy
except ImportError: install_requires.append('scipy')

try: import argparse
except ImportError: install_requires.append('argparse')

setup(
    name = "pyssim",
    version = "0.1",
    description = "Module for computing Structured Similarity Image Metric (SSIM) in Python",
    author = "Antoine Vacavant, Christopher Godfrey, Jeff Terrace",
    author_email = 'jterrace@gmail.com',
    platforms=["any"],
    license="BSD",
    install_requires=install_requires,
    url = "https://github.com/jterrace/pyssim",
    entry_points = {
        'console_scripts':[
            'pyssim = ssim.__main__:main'
        ]
    },
    packages = find_packages()
)
