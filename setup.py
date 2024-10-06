import os
from setuptools import setup, find_packages

setup(name='SunX',
      version='0.1',
      description='SunX: A Python Package for Modeling coronal loops',
      author='Biswajit Mondal',
      author_email='biswajit70mondal94@gmail.com',
      url='https://github.com/biswajitmb/SunX.git',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'matplotlib',
        'sunpy',
        'multiprocess',
        'yt',
        'configparser',
        'aiapy'
      ],
     )

