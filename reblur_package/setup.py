from setuptools import setup, find_packages
import unittest
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('reblur_package_lib', [
        'reblur_package_lib/cuda_arithmetic.cu',
        'reblur_package_lib/cuda_common.cu',
        'reblur_package_lib/cuda_geometry.cu',
        'reblur_package_lib/cuda_renderer.cu',
        'reblur_package_lib/cuda_renderer_flow.cpp',
        'reblur_package_lib/flow_blurrer.cpp',
        'reblur_package_lib/flow_forward_warp_mask.cpp',
        'reblur_package_lib/python_bind.cc',
        ]),
    ]

INSTALL_REQUIREMENTS = ['torch']

setup(
    description='reblur_package',
    author='Peidong Liu',
    author_email='peidong.liu@inf.ethz.ch',
    license='MIT License',
    version='0.0.1',
    name='reblur_package',
    packages=['reblur_package', 'reblur_package_lib'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)

