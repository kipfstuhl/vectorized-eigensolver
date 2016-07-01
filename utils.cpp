
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <sys/time.h>

#include "definitions.hpp"
#include "ComplexDouble.hpp"
#include "Toolbox.hpp"
#include "utils.h"
#define nullptr 0

using namespace std;

void loadPackages(int packages, int matrices, int n_coeffs, ComplexDouble* C)
{
    char numchar[3] = "";
    string numstring;
    string filePath;

    for (int package = 0; package < packages; package++)
    {
        sprintf(numchar, "%d", package + 1);
	string numstring(numchar);
        filePath = "testcases/f_coeffs_" + numstring + ".csv";
        cout << "loading package: " << filePath << endl;
        ifstream filestream(filePath.c_str());
	string line;

        for (int lineNumber = 0; lineNumber < matrices; lineNumber++)
        {
            getline(filestream, line);
            stringstream lineStream(line);
	    string cell;
            double coeffs[n_coeffs * 2];
            for (int coeffNumber = 0; coeffNumber < n_coeffs * 2; coeffNumber++)
            {
                getline(lineStream, cell, ',');
                coeffs[coeffNumber] = atof(cell.c_str());
            }
            for (int coeff_n = 0; coeff_n < n_coeffs; coeff_n++)
            {
                C[coeff_n + lineNumber * n_coeffs + package * matrices * n_coeffs] =
		  ComplexDouble(coeffs[coeff_n * 2], coeffs[coeff_n * 2 + 1]);
            }
        }
    }
}

void readMatrix(int dim, string file, double* array)
{
  // reads the binary file into array
  // the array has to be large enough

  int size = dim*dim * sizeof(double);
  ifstream is;
  is.open(file.c_str(), ios::binary);
  is.read((char *)array, size);
  is.close();

}

void readVector(int dim, string file, double* array)
{
  // reads the binary file into array
  // the array has to be large enough

  int size = dim * sizeof(double);
  ifstream is;
  is.open(file.c_str(), ios::binary);
  is.read((char *)array, size);
  is.close();
}

void readMatricesFromFile(int dim, int n_matrices, double* array, string file)
{
  int size = dim*dim * n_matrices * sizeof(double);
  ifstream is(file.c_str(), ios::binary);
  is.read((char *)array, size);
  is.close();
}

void rearrangeMatrices(int dim, int block_cnt, int thread_cnt, double* in, double* out)
{
  int dim2 = dim*dim;
  for(int block=0; block<block_cnt; ++block)
    {
      for(int thread=0; thread<thread_cnt; ++thread)
	{
	  for(int i=0; i<dim; ++i)
	    {
	      for(int j=0; j<dim; ++j)
		{
		  int ind_out = (i + j*dim) * thread_cnt + thread + block*thread_cnt*dim2;
		  int ind_in =  (i + j*dim) + dim2*(block*thread_cnt + thread);
		  out[ind_out] = in[ind_in];
		}
	    }
	}
    }
}

void arrangeMatricesAoS(int dim, int block_cnt, int thread_cnt, double *in, double *out)
{
  int dim2 = dim*dim;
  for(int block=0; block<block_cnt; ++block)
    {
      for(int thread=0; thread<thread_cnt; ++thread)
	{
	  for(int i=0; i<dim; ++i)
	    {
	      for(int j=0; j<dim; ++j)
		{
		  int ind_out = (i + j*dim) + dim2*(block*thread_cnt + thread);
		  int ind_in =  (i + j*dim) * thread_cnt + thread + block*thread_cnt*dim2;
		  out[ind_out] = in[ind_in];
		}
	    }
	}
    }
}

void repMatrixSoA(int dim, int block_cnt, int thread_cnt, double* matrix, double* out)
{
  // replicates the input matrix count times to out
  // in the data layout needed for the use of this library.
  // matrix has dimension dim

  int dim2 = dim*dim;

  for(int i=0; i<dim2; ++i)
    {
      for(int block=0; block<block_cnt; ++block)
	{
	  for(int thread=0; thread<thread_cnt; ++thread)
	    {
	      out[block*thread_cnt*dim2 + thread + i*thread_cnt] = matrix[i];
	    }
	}
    }
}

void repVectorSoA(int dim, int block_cnt, int thread_cnt, double *vec, double *out)
{
  for(int i=0; i<dim; ++i)
    {
      for(int block=0; block<block_cnt; ++block)
	{
	  for(int thread=0; thread<thread_cnt; ++thread)
	    {
	      out[block*thread_cnt*dim + thread + i*thread_cnt] = vec[i];
	    }
	}
    }
}

void call_zlahqr_new(int dim, int block_cnt, int thread_cnt, double* rE, double *iE,
		     double *rM, double *iM, double *rS, double *iS)
{
  int dim2 = dim*dim;
  int n_matrices = thread_cnt*block_cnt;
  int size = dim2 * n_matrices * sizeof(double);
  
#ifndef CUDA
  memset(rS, 0, size);
  memset(iS, 0, size);
  test_diag(iS, rS, 1.0, 0.0, dim, block_cnt, thread_cnt);

  zlahqr_new(rE, iE, rM, iM, rS, iS, dim, dim2, block_cnt, thread_cnt);
#else  // ndef CUDA
  int size_eig = dim * n_matrices * sizeof(double);
  
  double *rE_ = 0;
  double *iE_ = 0;
  double *rM_ = 0;
  double *iM_ = 0;
  double *rS_ = 0;
  double *iS_ = 0;

  cudaMalloc(&rE_, size_eig);
  cudaMalloc(&iE_, size_eig);

  cudaMalloc(&rM_, size);
  cudaMalloc(&iM_, size);
  cudaMemcpy(rM_, rM, size, cudaMemcpyHostToDevice);
  cudaMemcpy(iM_, iM, size, cudaMemcpyHostToDevice);
  
  cudaMalloc(&rS_, size);
  cudaMalloc(&iS_, size);
  cudaMemset(rS_, 0, size);
  cudaMemset(iS_, 0, size);
  test_diag<<<block_cnt, thread_cnt>>>(rS_, iS_, 1.0, 0.0, dim, block_cnt, thread_cnt);

  zlahqr_new<<<block_cnt, thread_cnt>>>(rE_, iE_, rM_, iM_, rS_, iS_, dim, dim2, block_cnt, thread_cnt);

  cudaMemcpy(rE, rE_, size_eig, cudaMemcpyDeviceToHost);
  cudaMemcpy(iE, iE_, size_eig, cudaMemcpyDeviceToHost);

  cudaMemcpy(rM, rM_, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(iM, iM_, size, cudaMemcpyDeviceToHost);

  cudaMemcpy(rS, rS_, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(iS, iS_, size, cudaMemcpyDeviceToHost);

  cudaFree(rE_);
  cudaFree(iE_);
  cudaFree(rM_);
  cudaFree(iM_);
  cudaFree(rS_);
  cudaFree(iS_);

#endif  // ndef CUDA

}

void call_zgemv(bool hermitian, int dim, int block_cnt, int thread_cnt, double *rM, double *iM,
		double *rx, double *ix, double *ry, double *iy,
		double rbeta, double ibeta)
{
 
#ifndef CUDA
  test_zgemv(hermitian, rM, iM, rx, ix, rbeta, ibeta, ry, iy, dim, block_cnt, thread_cnt);
#else  // ndef CUDA
  int dim2 = dim*dim;
  int n_matrices = thread_cnt * block_cnt;
  int sizeM = dim2 * n_matrices * sizeof(double);
  int sizeV = dim * n_matrices * sizeof(double);

  double *rM_ = 0;
  double *iM_ = 0;
  double *rx_ = 0;
  double *ix_ = 0;
  double *ry_ = 0;
  double *iy_ = 0;

  cudaMalloc(&rM_, sizeM);
  cudaMalloc(&iM_, sizeM);
  cudaMalloc(&rx_, sizeV);
  cudaMalloc(&ix_, sizeV);
  cudaMalloc(&ry_, sizeV);
  cudaMalloc(&iy_, sizeV);

  cudaMemcpy(rM_, rM, sizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(iM_, iM, sizeM, cudaMemcpyHostToDevice);
  cudaMemcpy(rx_, rx, sizeV, cudaMemcpyHostToDevice);
  cudaMemcpy(ix_, ix, sizeV, cudaMemcpyHostToDevice);
  cudaMemcpy(ry_, ry, sizeV, cudaMemcpyHostToDevice);
  cudaMemcpy(iy_, iy, sizeV, cudaMemcpyHostToDevice);

  test_zgemv<<<block_cnt, thread_cnt>>>(hermitian, rM_, iM_, rx_, ix_, rbeta, ibeta, ry_, iy_,
					dim, block_cnt, thread_cnt);
  
  cudaMemcpy(ry, ry_, sizeV, cudaMemcpyDeviceToHost);
  cudaMemcpy(iy, iy_, sizeV, cudaMemcpyDeviceToHost);

  cudaFree(rM_);
  cudaFree(iM_);
  cudaFree(rx_);
  cudaFree(ix_);
  cudaFree(iy_);
  cudaFree(ry_);
#endif	// ndef CUDA

  
}

void call_backsubstitution(int dim, int block_cnt, int thread_cnt,
			   double *rT, double *iT, double *rb, double *ib)
{
#ifndef CUDA
  test_backsubstitution(rT, iT, rb, ib, dim, block_cnt, thread_cnt);
#else  // ndef CUDA
  // code for CUDA goes here
#endif
}

void call_ztrevc(int dim, int block_cnt, int thread_cnt,
		 double *rT, double *iT, double *rR, double *iR)
{
  int dim2 = dim*dim;
  int n_matrices = block_cnt * thread_cnt;
#ifndef CUDA
  //  test_ident(rR, iR, dim, block_cnt, thread_cnt);
  double *rwork = new double[2*dim*n_matrices];
  double *iwork = new double[2*dim*n_matrices];
  test_ztrevc(rT, iT, rR, iR, rwork, iwork, dim, dim2, block_cnt, thread_cnt);
  delete[] rwork;
  delete[] iwork;
#else  // ndef CUDA
  int size = dim2 * n_matrices * sizeof(double);
  int size2 = dim * n_matrices * sizeof(double);

  double *rT_ = nullptr;
  double *iT_ = nullptr;
  double *rR_ = nullptr;
  double *iR_ = nullptr;
  double *rwork_ = nullptr;
  double *iwork_ = nullptr;

  cudaMalloc(&rT_, size);
  cudaMalloc(&iT_, size);
  cudaMalloc(&rR_, size);
  cudaMalloc(&iR_, size);
  cudaMalloc(&rwork_, size2);
  cudaMalloc(&rwork_, size2);
  
  cudaMemcpy(rT_, rT, size, cudaMemcpyHostToDevice);
  cudaMemcpy(iT_, iT, size, cudaMemcpyHostToDevice);
  cudaMemcpy(rR_, rR, size, cudaMemcpyHostToDevice);
  cudaMemcpy(iR_, iR, size, cudaMemcpyHostToDevice);

  //  test_ident<<<block_cnt, thread_cnt>>>(rR_, iR_, dim, block_cnt, thread_cnt);
  test_ztrevc<<<block_cnt, thread_cnt>>>(rT_, iT_, rR_, iR_, rwork_, iwork_, dim, dim2, block_cnt, thread_cnt);

  cudaMemcpy(rR, rR_, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(iR, iR_, size, cudaMemcpyDeviceToHost);

  cudaFree(rT_);
  cudaFree(iT_);
  cudaFree(rR_);
  cudaFree(iR_);
  cudaFree(rwork_);
  cudaFree(iwork_);
  
#endif	// ndef CUDA
}

void call_zgehrd(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau)
{
  int sizev = dim * thread_cnt * block_cnt;
#ifndef CUDA
  double *rwork = new double[sizev];
  double *iwork = new double[sizev];
  test_zgehrd(rA, iA, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);
  delete[] rwork;
  delete[] iwork;
#else  // ndef CUDA
  int sizem = dim*dim * thread_cnt * block_cnt * sizeof(double);
  sizev *= sizeof(double);
  
  double *rA_ = nullptr;
  double *iA_ = nullptr;
  double *rtau_ = nullptr;
  double *itau_ = nullptr;
  double *rwork_ = nullptr;
  double *iwork_ = nullptr;

  cudaMalloc(&rA_, sizem);
  cudaMalloc(&iA_, sizem);
  cudaMemcpy(rA_, rA, sizem, cudaMemcpyHostToDevice);
  cudaMemcpy(iA_, iA, sizem, cudaMemcpyHostToDevice);

  cudaMalloc(&rtau_, sizev);
  cudaMalloc(&itau_, sizev);
  cudaMalloc(&rwork_, sizev);
  cudaMalloc(&iwork_, sizev);

  test_zgehrd<<<block_cnt, thread_cnt>>>(rA_, iA_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  cudaMemcpy(rA, rA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iA, iA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(rtau, rtau_, sizev, cudaMemcpyDeviceToHost);
  cudaMemcpy(itau, itau_, sizev, cudaMemcpyDeviceToHost);

  cudaFree(rwork_);
  cudaFree(iwork_);
  cudaFree(rA_);
  cudaFree(iA_);
  cudaFree(rtau_);
  cudaFree(itau_);
#endif  // ndef CUDA
}

void call_zunghr(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau)
{
  int sizev = dim * thread_cnt * block_cnt;
#ifndef CUDA
  double *rwork = new double[sizev];
  double *iwork = new double[sizev];
  test_zunghr(rA, iA, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);
  delete[] rwork;
  delete[] iwork;
#else
  sizev *= sizeof(double);
  int sizem = dim*dim * thread_cnt * block_cnt;

  double *rA_, *iA_, *rtau_, *itau_, *rwork_, *iwork_;

  cudaMalloc(&rA_, sizem);
  cudaMemcpy(rA_, rA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&iA_, sizem);
  cudaMemcpy(iA_, iA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&rtau_, sizev);
  cudaMemcpy(rtau_, rtau, sizev, cudaMemcpyHostToDevice);
  cudaMalloc(&itau_, sizev);
  cudaMemcpy(itau_, itau, sizev, cudaMemcpyHostToDevice);
  cudaMalloc(&rwork_, sizev);
  cudaMalloc(&iwork_, sizev);

  test_zunghr<<<block_cnt, thread_cnt>>>(rA_, iA_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  cudaMemcpy(rA, rA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iA, iA_, sizem, cudaMemcpyDeviceToHost);

  cudaFree(rwork_);
  cudaFree(iwork_);
  cudaFree(rtau_);
  cudaFree(itau_);
  cudaFree(rA_);
  cudaFree(iA_);

#endif	// ndef CUDA
}

void hessenberg(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rQ, double *iQ)
{
  // calculate the Hessenberg form of A,
  // A = Q A Q^H

  int sizev = dim * thread_cnt * block_cnt;
  int sizem = dim*dim * thread_cnt * block_cnt * sizeof(double);
#ifndef CUDA
  double *rtau = new double[sizev];
  double *itau = new double[sizev];
  double *rwork = new double[sizev];
  double *iwork = new double[sizev];

  test_zgehrd(rA, iA, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  memcpy(rQ, rA, sizem);
  memcpy(iQ, iA, sizem);

  test_zunghr(rQ, iQ, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  delete[] rtau;
  delete[] itau;
  delete[] rwork;
  delete[] iwork;

#else
  sizev *= sizeof(double);

  double *rtau_, *itau_, *rwork_, *iwork_, *rA_, *iA_, *rQ_, *iQ_;
  cudaMalloc(&rA_, sizem);
  cudaMemcpy(rA_, rA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&iA_, sizem);
  cudaMemcpy(iA_, iA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&rQ_, sizem);
  cudaMalloc(&iQ_, sizem);
  cudaMalloc(&rtau_, sizev);
  cudaMalloc(&itau_, sizev);
  cudaMalloc(&rwork_, sizev);
  cudaMalloc(&iwork_, sizev);

  test_zgehrd<<<block_cnt, thread_cnt>>>(rA_, iA_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  cudaMemcpy(rQ_, rA_, sizem, cudaMemcpyDeviceToDevice);
  cudaMemcpy(iQ_, iA_, sizem, cudaMemcpyDeviceToDevice);

  test_zunghr<<<block_cnt, thread_cnt>>>(rQ_, iQ_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  cudaMemcpy(rA, rA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iA, iA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(rQ, rQ_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iQ, iQ_, sizem, cudaMemcpyDeviceToHost);

  cudaFree(rtau_);
  cudaFree(itau_);
  cudaFree(rwork_);
  cudaFree(iwork_);
  cudaFree(rA_);
  cudaFree(iA_);
  cudaFree(rQ_);
  cudaFree(iQ_);

#endif  // ndef CUDA
}

void eigensolver(int dim, int block_cnt, int thread_cnt, string real, string imag)
{
  // calculates all eigenvalues and eigenvectors of the matrices
  // read from the file

  // for CUDA there has to be done some additional work for allocations and copies
  int n_matrices = block_cnt * thread_cnt; // number of matrices
  int sizem = dim*dim * n_matrices;	   // number of double values to represent all matrices
  int sizev = dim * n_matrices;		   // number of double values to represent a vector

  double *rA = new double[sizem]; // input matrices
  double *iA = new double[sizem]; // input matricesn
  readMatricesFromFile(dim, n_matrices, rA, real);
  readMatricesFromFile(dim, n_matrices, iA, imag);
  
#ifndef CUDA

  double *rtau = new double[sizev];
  double *itau = new double[sizev];
  double *rwork = new double[2*sizev];
  double *iwork = new double[2*sizev];

  test_zgehrd(rA, iA, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  double *rR = new double[sizem]; // output for eigenvectors
  double *iR = new double[sizem]; // output for eigenvectors
  
  memcpy(rR, rA, sizem);
  memcpy(iR, iA, sizem);

  test_zunghr(rR, iR, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  double *rE = new double[sizev]; // output for eigenvalues
  double *iE = new double[sizev]; // output for eigenvalues
  
  zlahqr_new(rE, iE, rA, iA, rR, iR, dim, dim*dim, block_cnt, thread_cnt);

  test_ztrevc(rA, iA, rR, iR, rwork, iwork, dim, dim*dim, block_cnt, thread_cnt);

  delete[] rwork;
  delete[] iwork;
  delete[] rtau;
  delete[] itau;

#else  // ndef CUDA
  sizem *= sizeof(double);
  sizev *= sizeof(double);
  double *rA_, *iA_, *rtau_, *itau_, *rwork_, *iwork_;
  cudaMalloc(&rA_, sizem);
  cudaMalloc(&iA_, sizem);
  cudaMemcpy(rA_, rA, sizem, cudaMemcpyHostToDevice);
  cudaMemcpy(iA_, iA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&rtau_, sizev);
  cudaMalloc(&itau_, sizev);
  cudaMalloc(&rwork_, 2*sizev);
  cudaMalloc(&iwork_, 2*sizev);
  
  test_zgehrd<<<block_cnt, thread_cnt>>>(rA_, iA_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  double *rR_, *iR_;
  cudaMalloc(&rR_, sizem);
  cudaMalloc(&iR_, sizem);
  cudaMemcpy(&rR_, rA_, sizem, cudaMemcpyDeviceToDevice);
  cudaMemcpy(&iR_, iA_, sizem, cudaMemcpyDeviceToDevice);

  test_zunghr<<<block_cnt, thread_cnt>>>(rR_, iR_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);
  
  double *rE_, *iE_;
  cudaMalloc(&rE_, sizev);
  cudaMalloc(&iE_, sizev);

  zlahqr_new<<<block_cnt, thread_cnt>>>(rE_, iE_, rA_, iA_, rR_, iR_, dim, dim*dim, block_cnt, thread_cnt);

  test_ztrevc<<<block_cnt, thread_cnt>>>(rA_, iA_, rR_, iR_, rwork_, iwork_, dim, dim*dim, block_cnt, thread_cnt);

  cudaFree(rwork_);
  cudaFree(iwork_);
  cudaFree(rtau_);
  cudaFree(itau_);

  cudaFree(rE_);
  cudaFree(iE_);
  cudaFree(rA_);
  cudaFree(iA_);
  cudaFree(rR_);
  cudaFree(iR_);
  
					 
  
#endif	// ndef CUDA
  // eigenvectors are now in rR, iR
  // eigenvalues are Stored in rE, iE
  //
  // for proper output the arrays have to be converted in another data-layout

}

void eigensolver(int dim, int block_cnt, int thread_cnt,
		 double *rA, double *iA, double *rE, double *iE, double *rV, double *iV)
{
  // calculates all eigenvalues and eigenvectors of the matrices
  // in A, that are already in the right data layout
  // eigenvalues are returned in E, the corresponding vectors in V

  int n_matrices = block_cnt * thread_cnt; // number of matrices
  int sizem = dim*dim * n_matrices;	   // number of double values to represent all matrices
  int sizev = dim * n_matrices;		   // number of double values to represent a vector

  struct timeval t1, t2;
  double time;

#ifndef CUDA

  // double *rtau = new double[sizev];
  // double *itau = new double[sizev];
  // double *rwork = new double[2*sizev];
  // double *iwork = new double[2*sizev];
  double *rtau=0, *itau=0, *rwork=0, *iwork=0;
  posix_memalign((void **) &rtau, 4096, sizev * sizeof(double));
  posix_memalign((void **) &itau, 4096, sizev * sizeof(double));
  posix_memalign((void **) &rwork, 4096, 2*sizev * sizeof(double));
  posix_memalign((void **) &iwork, 4096, 2*sizev * sizeof(double));

  gettimeofday(&t1, NULL);

  test_zgehrd(rA, iA, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  memcpy(rV, rA, sizem * sizeof(double));
  memcpy(iV, iA, sizem * sizeof(double));

  test_zunghr(rV, iV, rtau, itau, rwork, iwork, dim, block_cnt, thread_cnt);

  zlahqr_new(rE, iE, rA, iA, rV, iV, dim, dim*dim, block_cnt, thread_cnt);

  test_ztrevc(rA, iA, rV, iV, rwork, iwork, dim, dim*dim, block_cnt, thread_cnt);

  gettimeofday(&t2, NULL);
  time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
  std::cout << time;

  // delete[] rwork;
  // delete[] iwork;
  // delete[] rtau;
  // delete[] itau;

  free(rtau);
  free(itau);
  free(rwork);
  free(iwork);
  
#else  // ndef CUDA
  sizem *= sizeof(double);
  sizev *= sizeof(double);
  double *rA_ = 0, *iA_ = 0, *rtau_ = 0, *itau_ = 0, *rwork_ = 0, *iwork_ = 0;
  cudaMalloc(&rA_, sizem);
  cudaMalloc(&iA_, sizem);
  cudaMemcpy(rA_, rA, sizem, cudaMemcpyHostToDevice);
  cudaMemcpy(iA_, iA, sizem, cudaMemcpyHostToDevice);
  cudaMalloc(&rtau_, sizev);
  cudaMalloc(&itau_, sizev);
  cudaMalloc(&rwork_, 2*sizev);
  cudaMalloc(&iwork_, 2*sizev);

  double *rV_, *iV_;
  cudaMalloc(&rV_, sizem);
  cudaMalloc(&iV_, sizem);

  double *rE_, *iE_;
  cudaMalloc(&rE_, sizev);
  cudaMalloc(&iE_, sizev);

  gettimeofday(&t1, NULL);
  
  test_zgehrd<<<block_cnt, thread_cnt>>>(rA_, iA_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);

  cudaMemcpy(rV_, rA_, sizem, cudaMemcpyDeviceToDevice);
  cudaMemcpy(iV_, iA_, sizem, cudaMemcpyDeviceToDevice);

  test_zunghr<<<block_cnt, thread_cnt>>>(rV_, iV_, rtau_, itau_, rwork_, iwork_, dim, block_cnt, thread_cnt);
  
  zlahqr_new<<<block_cnt, thread_cnt>>>(rE_, iE_, rA_, iA_, rV_, iV_, dim, dim*dim, block_cnt, thread_cnt);

  test_ztrevc<<<block_cnt, thread_cnt>>>(rA_, iA_, rV_, iV_, rwork_, iwork_, dim, dim*dim, block_cnt, thread_cnt);

  cudaThreadSynchronize();
  gettimeofday(&t2, NULL);
  time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
  std::cout << time;

  cudaFree(rwork_);
  cudaFree(iwork_);
  cudaFree(rtau_);
  cudaFree(itau_);

  cudaMemcpy(rE, rE_, sizev, cudaMemcpyDeviceToHost);
  cudaMemcpy(iE, iE_, sizev, cudaMemcpyDeviceToHost);
  cudaFree(rE_);
  cudaFree(iE_);

  cudaMemcpy(rA, rA_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iA, iA_, sizem, cudaMemcpyDeviceToHost);
  cudaFree(rA_);
  cudaFree(iA_);

  cudaMemcpy(rV, rV_, sizem, cudaMemcpyDeviceToHost);
  cudaMemcpy(iV, iV_, sizem, cudaMemcpyDeviceToHost);
  cudaFree(rV_);
  cudaFree(iV_);
  
					 
  
#endif	// ndef CUDA
  // eigenvectors are now in rR, iR
  // eigenvalues are Stored in rE, iE
  //
  // for proper output the arrays have to be converted in another data-layout

}
