from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("libmesh", sources=["triangle_hash.pyx"], include_dirs=[numpy.get_include()])
]

setup(
      name='libmesh',
      ext_modules=cythonize(extensions),
)
