
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "python_interface",
    cmdclass = {"build_ext": build_ext},
    ext_modules = [
    Extension(name="python_interface", sources=["python_interface.pyx"],
              libraries = ["m"], language="c++",
              extra_compile_args=['-fopenmp', '-O3'],
              extra_link_args=['-fopenmp'],
              define_macros=[('OPENMP', None)])
    ],
    )
