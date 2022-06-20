"""Setup script for pyssim."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='pyssim',
    version='0.5',
    description=('Module for computing Structured Similarity Image Metric '
                 '(SSIM) in Python'),
    author='Antoine Vacavant, Christopher Godfrey, Jeff Terrace',
    author_email='jterrace@gmail.com',
    platforms=['any'],
    license='MIT License',
    install_requires=['numpy', 'pillow', 'scipy'],
    url='https://github.com/jterrace/pyssim',
    entry_points={
        'console_scripts': [
            'pyssim = ssim.__main__:main'
        ]
    },
    packages=find_packages()
)
