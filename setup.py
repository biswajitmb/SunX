import os
from setuptools import setup, find_packages

setup(name='arsim',
      version='0.1',
      description='arsim: A Python Package for Modeling coronal loops',
      author='Biswajit Mondal',
      author_email='biswajit.mondal@nasa.gov',
      #url='https://github.com/biswajitmb/DarpanX/tree/objOriented/objectOriented/python',
      packages=find_packages()
     )

#print("%% SimAR_message : Don't delete the directory '"+os.getcwd()+"'")
