//============================================================================
// Name        : zlahqr.cpp
// Author      : M. Presenhuber, M. Liebmann
// Version     : 1.0
// Copyright   : University of Graz
// Description : ZLAQHRV-Algorithm
//============================================================================


#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sys/time.h>

#ifdef OPENMP
#include "omp.h"
#endif

#ifdef OPENACC
#include "openacc.h"
#endif

#ifdef CUDA

#define cuda __host__ __device__
#define global __global__
#else
#define cuda
#define global
#endif

#ifdef LAPACK
#define ZLAHQR zlahqr__
#else
#define ZLAHQR zlahqr_
#endif

#include "definitions.hpp"
#include "ComplexDouble-def.hpp"
#include "Toolbox-def.hpp"
#include "utils.h"

using namespace std;

int main(int argc, char** argv)
{
  int dim = 10;
  int dim2 = dim*dim;

  string realFile("testcases/real.bin");
  string imagFile("testcases/imag.bin");
  int n_matrices = 65536 * 16;
  // 65536 = 2**16
  int thread_cnt, block_cnt;
  int copysize = dim2 * n_matrices * sizeof(double);

#ifndef CUDA
  thread_cnt = 1;
  block_cnt = n_matrices / 1;
#else  // CUDA
  thread_cnt = 32;
  block_cnt = n_matrices / 32;
  cudaSetDevice(2);
  cudaFree(0);

/*
  cudaFuncSetCacheConfig(test_zgehrd, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(test_zunghr, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(zlahqr_new, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(test_ztrevc, cudaFuncCachePreferL1);
*/
#endif	// CUDA

  if(n_matrices != block_cnt * thread_cnt)
    {
      cout << "The number of matrices and the parameters block_cnt, thread_cnt do not match!" << endl;
      return 1;
    }

  int memorySize;
  memorySize = 4 * dim2*n_matrices;
  memorySize += 2 * dim*n_matrices;
  cout << "Memory in use for matrices, eigenvectors and eigenvalues: " << (double) 8*memorySize/1024/1024/1024 <<
    " GB" << endl;
  memorySize += 2 * dim*n_matrices;
  memorySize += 4 * dim*n_matrices;
  cout << "Total memory in use: " << (double) 8*memorySize/1024/1024/1024 << " GB" << endl;
  

#ifdef CUDA
  size_t freemem;
  size_t totalmem;
  cudaMemGetInfo(&freemem, &totalmem);
  if(freemem < sizeof(double)*memorySize)
    {
      cout << "GPU device has not enough free memory" << endl;
      cout << "Free: " << (double) freemem/1024/1024/1024 << " GB" << endl;
      return 1;
    }
#endif	// CUDA
  
  //double *realA = new double[dim2 * thread_cnt*block_cnt]();
  //double *imagA = new double[dim2 * thread_cnt*block_cnt]();
  double *realA=0, *imagA=0;
  posix_memalign((void **) &realA, 4096, dim2 * n_matrices * sizeof(double));
  posix_memalign((void **) &imagA, 4096, dim2 * n_matrices * sizeof(double));

  readMatricesFromFile(dim, n_matrices, realA, realFile);
  readMatricesFromFile(dim, n_matrices, imagA, imagFile);
  

  // double *rVec = new double[dim2 * n_matrices]();
  // double *iVec = new double[dim2 * n_matrices]();
  // double *rVal = new double[dim * n_matrices]();
  // double *iVal = new double[dim * n_matrices]();
  double *rVec=0, *iVec=0, *rVal=0, *iVal=0;
  posix_memalign((void **) &rVec, 4096, dim2 * n_matrices * sizeof(double));
  posix_memalign((void **) &iVec, 4096, dim2 * n_matrices * sizeof(double));
  posix_memalign((void **) &rVal, 4096, dim * n_matrices * sizeof(double));
  posix_memalign((void **) &iVal, 4096, dim * n_matrices * sizeof(double));
  
  struct timeval t1, t2;
  double time;
/*
  gettimeofday(&t1, NULL);

  eigensolver(dim, block_cnt, thread_cnt, realA, imagA, rVal, iVal, rVec, iVec);

  gettimeofday(&t2, NULL);

  time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
  cout << "Elapsed time: " <<  time << " s" << endl;
*/
  
  // further Testing with different parameters
  //double *tempreal = new double[dim2 * thread_cnt*block_cnt];
  //double *tempimag = new double[dim2 * thread_cnt*block_cnt];
  double *tempreal=0, *tempimag=0;
  posix_memalign((void **) &tempreal, 4096, dim2 * n_matrices * sizeof(double));
  posix_memalign((void **) &tempimag, 4096, dim2 * n_matrices * sizeof(double));

  readMatricesFromFile(dim, n_matrices, tempreal, realFile);
  readMatricesFromFile(dim, n_matrices, tempimag, imagFile);

#ifdef OPENMP
  thread_cnt = 1;
  block_cnt = n_matrices / thread_cnt;
#else  // OPENMP
#ifdef CUDA
  thread_cnt = 32;
  block_cnt = n_matrices / thread_cnt;
#endif	// CUDA
#endif	// OPENMP
  /*
  cout << "#Blocks  VectorSize  Time" << endl;
  for(int i=0; i<6; ++i)
    {
      if(n_matrices != block_cnt * thread_cnt)
	{
	  cout << "Parameter mismatch: " << endl;
	  cout << "block_cnt:  " << block_cnt << endl;
	  cout << "thread_cnt: " << thread_cnt << endl;
	  cout << "n_matrices: " << n_matrices << endl;
	  return 1;
	}


      memcpy(realA, tempreal, dim2*n_matrices * sizeof(double));
      memcpy(imagA, tempimag, dim2*n_matrices * sizeof(double));
      

      cout << block_cnt << " " << thread_cnt << " ";
      gettimeofday(&t1, NULL);
      eigensolver(dim, block_cnt, thread_cnt, realA, imagA, rVal, iVal, rVec, iVec);
      gettimeofday(&t2, NULL);
      //cout << "block_cnt:    " << block_cnt << endl;
      //cout << "thread_cnt:   " << thread_cnt << endl;
      time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
      //cout << "Elapsed time: " << time << " s" << endl;
      cout << endl;
      
      block_cnt /= 2;
      thread_cnt *= 2;
      
      //cout << endl;
      //for(int i = 0; i < dim; i++) cout << rVal[i] << " " << iVal[i] << " | ";
      //cout << endl;
    }
  */

  cout << endl << endl;

  //int n_testing = 16384;	// from previous results
  int n_testing = 13312;
#ifdef OPENMP
  thread_cnt = 4;
  block_cnt = n_testing / thread_cnt;
#else  // OPENMP
#ifdef CUDA
  thread_cnt = 512;
  block_cnt = n_testing / thread_cnt;
#endif	// CUDA
#endif	// OPENMP
  dim = 10;
  dim2 = dim*dim;

  cout << "# n_matrices:   " << n_testing << endl;
  cout << "# block_cnt:    " << block_cnt << endl;
  cout << "# thread_cnt:   " << thread_cnt << endl;
  cout << "# Time   Dimension" << endl;

  for(int i=0; i<70; ++i)
    {
      if(n_testing != thread_cnt*block_cnt)
	{
	  cout << "Parameter mismatch!" << endl;
	  cout << "block_cnt:  " << block_cnt << endl;
	  cout << "thread_cnt: " << thread_cnt << endl;
	  cout << "n_matrices: " << n_matrices << endl;
	  return 1;
	}
      if(dim2*n_testing*sizeof(double) > copysize)
	{
	  cout << "Dimension got too large, not enough memory!" << endl;
	  cout << "dim: " << dim << endl;
	  return 1;
	}

      memcpy(realA, tempreal, copysize);
      memcpy(imagA, tempimag, copysize);

      cout << dim << " ";
      gettimeofday(&t1, NULL);
      eigensolver(dim, block_cnt, thread_cnt, realA, imagA, rVal, iVal, rVec, iVec);
      gettimeofday(&t2, NULL);
      time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;

      cout << " # " << block_cnt << " " << thread_cnt << endl;
      //cout << "Dimension:  " << dim << endl;
      //cout << "n_matrices: " << n_matrices << "\t block_cnt: " << block_cnt <<
      //  "\t thread_cnt: " << thread_cnt << endl;
      //cout << "Elapsed time: " << time << " s" << endl;

      //n_matrices /= 4;
      //block_cnt /= 4;
      //dim *= 2;
      dim += 1;
      dim2 = dim*dim;
      
      }

  
  // delete[] tempreal;
  // delete[] tempimag;
  free(tempreal);
  free(tempimag);


  // delete[] realA;
  // delete[] imagA;
  // delete[] rVec;
  // delete[] iVec;
  // delete[] rVal;
  // delete[] iVal;
  free(realA);
  free(imagA);
  free(rVec);
  free(iVec);
  free(rVal);
  free(iVal);
  
  return 0;
}
