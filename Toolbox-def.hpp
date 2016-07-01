
#ifndef TOOLBOX_DEF_HPP_
#define TOOLBOX_DEF_HPP_

cuda void scale_rc(double *rV, double *iV, double zr, double zi, int I, int I1, int I2, int dim, int n);

global void test_zgemv(bool hermitian, double *rA, double *iA, double *rx, double *ix, double rbeta, double ibeta,
		       double *ry, double *iy, int dim, int block_cnt, int thread_cnt);

global void test_ztrevc(double *rV, double *iV, double *rZ, double *iZ, double *rwork, double *iwork,
			int dim, int dim2, int block_cnt, int thread_cnt);

global void zlahqr_new(double* rE, double* iE, double* rV, double* iV, double* rZ, double* iZ,
		       int dim, int dim2, int block_cnt, int thread_cnt);

global void test_ident(double* rZ, double* iZ, int dim, int block_cnt, int thread_cnt);

global void test_zgehrd(double * rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
			int dim, int block_cnt, int thread_cnt);

global void test_zunghr(double *rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
			int dim, int block_cnt, int thread_cnt);

#endif	// TOOLBOX_DEF_HPP_
