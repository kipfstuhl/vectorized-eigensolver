
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = "python_interface",
    cmdclass = {"build_ext": build_ext},
    ext_modules = [
    Extension(name="python_interface", sources=["python_interface.pyx"],
              libraries = ["m"], language="c++",
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp'],
              define_macros=[('OPENMP', None)])
    ],
    )
