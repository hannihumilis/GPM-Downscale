#!/usr/bin/env python

from setuptools import setup

setup(name='GPM-Downscale',
      version='1.0',
      description='Downscaling of precipitation (GPM) for Iberia',
      author='Johanna Roschke',
      author_email='roschke.johanna@web.de',
      url='https://hannihumilis/GPM-Downscale',
      license='MIT',
      packages=[],
      install_requires=[
          'numpy',
          'xarray',
          'scipy',
          'pandas',
          'scikit-learn',
          'georasters',
          'rasterio',
          'os',
          'fiona',

      ],
     )
