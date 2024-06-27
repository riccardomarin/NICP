from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("libvoxelize", sources=["voxelize.pyx"], include_dirs=[numpy.get_include(), "./utils/libvoxelize"])
]

setup(
      name='libvoxelize',
      ext_modules=cythonize(extensions),
)
