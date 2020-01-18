from setuptools import setup, find_packages
import unittest
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('correlation_package_lib', [
        'correlation_package_lib/corr_cuda_kernel.cu',
        'correlation_package_lib/corr_cuda.cpp',
        'correlation_package_lib/python_bind.cc',
        ]),
    ] 

INSTALL_REQUIREMENTS = ['torch']

setup(
    description='correlation_package',
    author='Peidong Liu',
    author_email='peidong.liu@inf.ethz.ch',
    license='MIT License',
    version='0.0.1',
    name='correlation_package',
    packages=['correlation_package', 'correlation_package_lib'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)


