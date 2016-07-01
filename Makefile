# //============================================================================
# // Name        : Makefile
# // Author      : M. Presenhuber, M. Liebmann
# // Version     : 1.0
# // Copyright   : University of Graz
# // Description : ZLAQHRV-Algorithm
# //============================================================================

#FLAGS = -lm -O3 -std=c++11
FLAGS = -O3
#CUDA_FLAGS = -D_FORCE_INLINES

makefile:
all:	gpp cuda
gpp:	#utils.o
#	g++ -O3 -march=native -o zlahqr zlahqr.cpp -Wall -lm
	g++ -O3 -march=corei7-avx -fopenmp -o zlahqr-i7 zlahqr.cpp utils.cpp -Wall -lm -DOPENMP
	g++ -O3 -march=native -fopenmp -o zlahqr-omp zlahqr.cpp utils.cpp -Wall -lm -DOPENMP
#	g++ -O3 -march=native -fopenmp -o zlahqr zlahqr.cpp -Wall -lm -llapack -DOPENMP -DLAPACK
#	g++ -O3 -march=native -fopenmp -o zlahqr zlahqr.cpp -Wall -lm -llapack -DOPENMP -DLAPACK -DZGEEV

cuda:
#	nvcc -O3 -x cu -arch=sm_20 -fmad=false -o zlahqr zlahqr.cpp -lm -DCUDA
#	nvcc -O3 -x cu -arch=sm_21 -o zlahqr zlahqr.cpp -lm -DCUDA
	nvcc -O3 -x cu -arch=sm_20 -o zlahqr-cuda zlahqr.cpp utils.cpp -DCUDA
	nvcc -O3 -x cu -arch=sm_35 -o zlahqr-cuda-k20 zlahqr.cpp utils.cpp -DCUDA

cuda-inline:
	nvcc $(FLAGS) $(CUDA_FLAGS) -x cu -arch=sm_20 -o zlahqr-cuda zlahqr.cpp utils.cpp -DCUDA -D_FORCE_INLINES

kepler:
	nvcc -O3 -x cu -arch=sm_35 -o zlahqr-kepler zlahqr.cpp utils.cpp -DCUDA

icpc:
	icpc -O3 -mavx -qopenmp -o zlahqr zlahqr.cpp -Wall -fp-model precise -lm -DOPENMP

pgcpp:
	pgc++ -O4 -acc -Minfo=accel -ta=tesla:cc20,cc35 -o zlahqr zlahqr.cpp -lm -DOPENACC

mkl:
	icpc -O3 -march=native -qopenmp -o zlahqr zlahqr.cpp -Wall -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -DOPENMP -DMKL

micoff:
	icpc -O3 -march=native -qopenmp -o zlahqr zlahqr.cpp -Wall -lm -offload-attribute-target=mic -DMIC -DOPENMP

mic:
	icpc -O3 -mmic -qxopenmp -o zlahqr zlahqr.cpp -Wall -lm -DOPENMP

gpp-debug:	utils-debug.o
	g++ -g -march=native -o zlahqr utils-debug.o zlahqr.cpp -Wall -lm

utils.o:	utils.cpp utils.h Toolbox.hpp
	g++ $(FLAGS) -c -fopenmp -march=native -Wall -o utils.o utils.cpp -DOPENMP

libutilscu.a:	utils.cpp utils.h Toolbox.hpp
	nvcc $(FLAGS) $(CUDA_FLAGS) -c --lib -x cu -arch=sm_20 -o libutilscu.a utils.cpp -DCUDA

utils-debug.o:	utils.cpp utils.h Toolbox.hpp
	g++ -c -g -march=native -o utils-debug.o -Wall -std=c++11 utils.cpp

python:
	rm -rf build
	rm -f python_interface.cpp
	rm -f python_interface.so
	python setup.py build_ext --inplace
#	/share/apps/python/bin/python setup.py python_interface.pyx utils.cpp




