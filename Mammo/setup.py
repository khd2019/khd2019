#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2
from distutils.core import setup
import setuptools

setup(
    name='KHD_mammon_train',
    version='1.0',
    install_requires=['scikit-learn']
)
