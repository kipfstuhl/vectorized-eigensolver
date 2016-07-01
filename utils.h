
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>
#include <cstring>

#ifdef OPENMP
#include "omp.h"
#endif

#ifdef CUDA

#define cuda __host__ __device__
#define global __global__

#else  // CUDA

#define cuda
#define global

#endif

#ifdef LAPACK
#define ZLAHQR zlahqr__
#else
#define ZLAHQR zlahqr_
#endif

//#include "definitions.hpp"

void loadPackages(int packages, int matrices, int n_coeffs, ComplexDouble *C);

void readMatrix(int dim, std::string file, double *array);
void readVector(int dim, std::string file, double *array);
void readMatricesFromFile(int dim, int n_matrices, double* array, std::string file);

void repMatrixSoA(int dim, int block_cnt, int thread_cnt, double* matrix, double* out);
void repVectorSoA(int dim, int block_cnt, int thread_cnt, double *vec, double *out);

void call_zlahqr_new(int dim, int block_cnt, int thread_cnt,
		     double* rE, double *iE, double *rM, double *iM, double *rS, double *iS);

void call_zgemv(bool hermitian, int dim, int block_cnt, int thread_cnt,
		double *rM, double *iM,	double *rx, double *ix, double *ry, double *iy,
		double rbeta, double ibeta);

void call_ztrevc(int dim, int block_cnt, int thread_cnt,
		 double *rT, double *iT, double *rR, double *iR);

void call_zgehrd(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau);

void call_zunghr(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau);

void hessenberg(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rQ, double *iQ);

void eigensolver(int dim, int block_cnt, int thread_cnt, std::string real, std::string imag);

void eigensolver(int dim, int block_cnt, int thread_cnt,
		 double *rA, double *iA, double *rE, double *iE, double *rV, double *iV);

#endif	// UTILS_HPP_
