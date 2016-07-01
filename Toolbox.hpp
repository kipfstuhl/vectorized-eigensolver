//============================================================================
// Name        : Toolbox.hpp
// Author      : J. Kipfstuhl, M. Presenhuber, M. Liebmann
// Version     : 2.0
// Copyright   : University of Graz
// Description : ZLAQHRV-Algorithm
//============================================================================

#ifndef TOOLBOX_HPP_
#define TOOLBOX_HPP_

//#include "ComplexDouble.hpp"

cuda void scale_rc(double *rV, double *iV, double zr, double zi, int I, int I1, int I2, int dim, int n)
{
    int off = 1 + dim, j;
    j = (I + (I + 1) * dim - off) * n;
    for (int i = 0; i < I2 - I; ++i)
    {
        cmul(rV[j], iV[j], zr, zi);
        j += dim * n;
    }
    j = (I1 + I * dim - off) * n;
    for (int i = 0; i < I - I1; ++i)
    {
        cmulc(rV[j], iV[j], zr, zi);
        j += n;
    }
}

cuda void backsubstitution(int n, double *rA, double *iA, double *rx, double *ix,
			   int dim, int n_matrices)
{
  // performs simple backsubstituion for A*x=b, where
  // A is a n by n upper triangular matrix
  // b is a n-dimensional vector
  // x is to be calculated
  // on entry the right-hand side is stored in x
  // this is overwritten with the solution of the system
  //
  // norm is a n-dim array containing the norms of the off-diagonal part of columns of A
  //
  // dim is the dimension of the complete matrix; needed for the data structure
  // n is the dimension of the working part of A
  // 
  // this is only a simple form of the algorithm

  int off = n_matrices + dim * n_matrices; // offset for matrices, used for ease of comparison with Fortran code
  int offv = n_matrices;		   // offset for vector
  int iv;				   // index for vector
  int im;				   // index for matrix
  double a1r, a1i, t1r, t1i, x1r, x1i;

  /* not needed for this simple version
  double smlnum = DBL_MIN/DBL_EPSILON;
  double bignum = 1.0/smlnum;	// use DBL_MAX instead????
  double scale = 1.0;
  double tscal;
  */
  
  // ommitted optimizations from BLAS:
  // there is for example checked wether
  // x(j) or rx[iv]+i*ix[iv] is zero
  // and if the diagonal of A is just ones
  //
  // for use on a GPU this version without extra checks
  // should be better, due to less thread divergence

  /*
    omitted, as this code is just for handling ill conditioned matrices
    the other parts of this are not implemented
  int imax=1;
  double tmax = norm[0];
  for(int i=1; i<=n; ++i)
    {
      iv = i * n_matrices - offv;
      double temp = norm[iv];
      if(temp > tmax)
	{
	  tmax = temp;
	  imax = i;
	}
    }
  if(tmax <= bignum*0.5)
    {
      tscal = 1.0;
    }
  else
    {
      tscal = 0.5 / (smlnum * tmax);
      // scale norm by tscal
      for(int i=1; i<= n; ++i)
	{
	  iv = i * n_matrices - offv;
	  norm[iv] *= tscal;
	}
    }
  */

  // don't calculate grow and omit check grow*tscal > smlnum ?
  
  
  
  // essentially loop 20 of BLAS Routine ztrs
  for(int j=n; j>=1; --j)
    {
      im = (j + j * dim) * n_matrices - off;
      iv = j * n_matrices - offv;
      t1r = rx[iv];
      t1i = ix[iv];
      a1r = rA[im];
      a1i = iA[im];

      cdiv(t1r, t1i, a1r, a1i);
      rx[iv] = t1r;
      ix[iv] = t1i;

      im = (j-1 + j * dim) * n_matrices - off;
      for(int i=j-1; i>=1; --i)
	{
	  //im = (i + j * dim) * n_matrices - off;
	  iv = i * n_matrices - offv;
	  x1r = rx[iv];
	  x1i = ix[iv];
	  a1r = rA[im];
	  a1i = iA[im];

	  cmul(a1r, a1i, t1r, t1i);
	  csub(x1r, x1i, a1r, a1i);

	  rx[iv] = x1r;
	  ix[iv] = x1i;
	  im -= n_matrices;
	}
    }
}

cuda void single_shift_qr_schur(bool wantT, bool wantZ,
				double *rV, double *iV, double *rZ, double *iZ,
				int I, int L, int M, int I1, int I2, double v0r, double v0i, double v1r, double v1i,
				int dim, int n_matrices)
{
  // calculates one single-shift QR step
  // wantT:  is Schur matrix T required?
  // wantZ:  is matrix of Schur vectors Z required?
  // rV, iV: the Hessenberg matrix
  // rZ, iZ: the matrix of schur vectors

  
    int off = n_matrices + dim * n_matrices;
    int ILOZ = 1, IHIZ = dim;
    int nz = IHIZ - ILOZ + 1;
    double t1r, t1i, t2;
    double h1r, h1i, h2r, h2i, zr, zi, d;
    double z1r, z1i, z2r, z2i;
    int i1, i2, i3;

    for (int K = M; K <= I - 1; ++K) // loop 120, line 454
    {
      // The first iteration of this loop determines a reflection G
      // from the vector V and applies it from left and right to
      // thus creating a nonzero bulge below the subdiagonal.
      // 
      // Each subsequent iteration determines a reflection G to
      // restore the Hessenberg form in the (K-1)th column, and thus
      // chases the bulge one step toward the bottom of the active
      // submatrix.
      // 
      // v1 is always real (i.e. v1i=0) before the call to ZLARFG, and hence
      // after the call t2 ( = t1*v1 ) is also real.

      
        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            v0r = rV[i1];
            v0i = iV[i1];
            v1r = rV[i2];
            v1i = iV[i2];
        }
	
	// call to zlarfg
	// calculates a elementary Householder reflector
	// t1 <-> tau
        if (v1r == 0.0 && v1i == 0.0 && v0i == 0.0)
        {
            t1r = 0.0;
            t1i = 0.0;
        }
        else
        {
            double norm, beta, rsafmn, safmin;
            safmin = DBL_MIN / (DBL_EPSILON * 0.5);

	    // reordering of some computationsn
#ifdef MKL
            norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
            norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            int KNT = 0;
            if (norm < safmin)
            {
                rsafmn = 1.0 / safmin;
                do
                {
                    KNT = KNT + 1;
                    cmul(v1r, v1i, rsafmn);
                    cmul(v0r, v0i, rsafmn);
                    norm *= rsafmn;
                }
                while (norm < safmin);

#ifdef MKL
                norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
                norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            }
            beta = copysign(norm, v0r); // beta has a different sign than in LAPACK
            zr = v0r + beta;
            zi = v0i;
            t1r = zr;
            t1i = zi;

#ifdef MKL
            cdivex(1.0, 0.0, t1r, t1i, zr, zi);
#else
            cinv(zr, zi);
#endif

            cmul(v1r, v1i, zr, zi);
            cdiv(t1r, t1i, beta);

            for (int i = 0; i < KNT; ++i)
                beta *= safmin;
            v0r = -beta;
            v0i = 0.0;
        } // end of zlarfg
	// reflector is calculated

        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            rV[i1] = v0r;
            iV[i1] = v0i;
            rV[i2] = 0.0;
            iV[i2] = 0.0;
        }

        t2 = t1r * v1r - t1i * v1i;

	// Apply G from the left to transform the rows of
	// the matrix in columns K to I2
        i1 = (K + K * dim) * n_matrices - off;
        i2 = i1 + n_matrices;
        for (int j = K; j <= I2; ++j) // loop 80
        {
            h1r = rV[i1];
            h1i = iV[i1];
            zr = h1r;
            zi = h1i;	    
            cmulc(zr, zi, t1r, t1i);
            h2r = rV[i2];
            h2i = iV[i2];
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmul(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += dim * n_matrices;
            i2 += dim * n_matrices;
        }

	// Apply G from the right to transform the columns of the
	// matrix in rows I1 to min(K+2, I)
        i1 = (I1 + K * dim) * n_matrices - off;
        i2 = i1 + dim * n_matrices;
        i3 = (K + 2) < I ? (K + 2) : I;
        for (int j = I1; j <= i3; ++j)
        {
            h1r = rV[i1];
            h1i = iV[i1];
            zr = h1r;
            zi = h1i;
            cmul(zr, zi, t1r, t1i);
	    h2r = rV[i2];
            h2i = iV[i2];
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmulc(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += n_matrices;
            i2 += n_matrices;
        }

	// accumulate transformations in the matrix Z
	if(wantZ)
	  {
	    i1 = (ILOZ + K * dim) * n_matrices - off;
	    i2 = i1 + dim * n_matrices;
	    for(int j = ILOZ; j <= IHIZ; ++j)
	      {
		z1r = rZ[i1];
		z1i = iZ[i1];
		zr = z1r;
		zi = z1i;
		cmul(zr, zi, t1r, t1i);
		z2r = rZ[i2];
		z2i = iZ[i2];
		cadd(zr, zi, t2 * z2r, t2 * z2i);
		csub(z1r, z1i, zr, zi);
		cmulc(zr, zi, v1r, v1i);
		csub(z2r, z2i, zr, zi);
		rZ[i1] = z1r;
		iZ[i1] = z1i;
		rZ[i2] = z2r;
		iZ[i2] = z2i;
		i1 += n_matrices;
		i2 += n_matrices;
	      }
	  }
	// special case is omitted, bad for vectorized performance
    }

    // ensure H(I,I-1) is real
    i1 = (I + (I - 1) * dim) * n_matrices - off;
    zi = iV[i1];
    if (zi != 0.0)
    {
        zr = rV[i1];
        d = cabs(zr, zi);
        cdiv(zr, zi, d);
        zi = -zi;
        rV[i1] = d;
        iV[i1] = 0.0;
        scale_rc(rV, iV, zr, zi, I, I1, I2, dim, n_matrices);

	if(wantZ)
	  {

	    //int off_scale = 1 + dim;
	    //int j;
	    i1 = (ILOZ + I * dim) * n_matrices - off;
	    for (int i = 0; i < nz; ++i)
	      {
		z1r = rZ[i1];
		z1i = iZ[i1];
		cmulc(z1r, z1i, zr, zi);
		rZ[i1] = z1r;
		iZ[i1] = z1i;
		i1 += n_matrices;
	      }
	  }
	    
    }

}


cuda void single_shift_qr(double *rV, double *iV, int I, int L, int M, int I1, int I2,
			  double v0r, double v0i, double v1r, double v1i, int dim, int n_matrices)
{
    int off = n_matrices + dim * n_matrices;
    double t1r, t1i, t2;
    double h1r, h1i, h2r, h2i, zr, zi, d;
    int i1, i2, i3;

    for (int K = M; K <= I - 1; ++K)
    {
        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            v0r = rV[i1];
            v0i = iV[i1];
            v1r = rV[i2];
            v1i = iV[i2];
        }

        if (v1r == 0.0 && v1i == 0.0 && v0i == 0.0)
        {
            t1r = 0.0;
            t1i = 0.0;
        }
        else
        {
            double norm, beta, rsafmn, safmin;
            safmin = DBL_MIN / (DBL_EPSILON * 0.5);

#ifdef MKL
            norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
            norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            int KNT = 0;
            if (norm < safmin)
            {
                rsafmn = 1.0 / safmin;
                do
                {
                    KNT = KNT + 1;
                    cmul(v1r, v1i, rsafmn);
                    cmul(v0r, v0i, rsafmn);
                    norm *= rsafmn;
                }
                while (norm < safmin);

#ifdef MKL
                norm = dabs3(v0r, v0i, dabs2(v1r, v1i));
#else
                norm = sqrt(v0r * v0r + v0i * v0i + v1r * v1r + v1i * v1i);
#endif

            }
            beta = copysign(norm, v0r);
            zr = v0r + beta;
            zi = v0i;
            t1r = zr;
            t1i = zi;

#ifdef MKL
            cdivex(1.0, 0.0, t1r, t1i, zr, zi);
#else
            cinv(zr, zi);
#endif

            cmul(v1r, v1i, zr, zi);
            cdiv(t1r, t1i, beta);

            for (int i = 0; i < KNT; ++i)
                beta *= safmin;
            v0r = -beta;
            v0i = 0.0;
        }

        if (K > M)
        {
            i1 = (K + (K - 1) * dim) * n_matrices - off;
            i2 = i1 + n_matrices;
            rV[i1] = v0r;
            iV[i1] = v0i;
            rV[i2] = 0.0;
            iV[i2] = 0.0;
        }

        t2 = t1r * v1r - t1i * v1i;

        i1 = (K + K * dim) * n_matrices - off;
        i2 = i1 + n_matrices;
        for (int j = K; j <= I2; ++j)
        {
            h1r = rV[i1];
            h1i = iV[i1];
            h2r = rV[i2];
            h2i = iV[i2];
            zr = h1r;
            zi = h1i;
            cmulc(zr, zi, t1r, t1i);
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmul(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += dim * n_matrices;
            i2 += dim * n_matrices;
        }

        i1 = (I1 + K * dim) * n_matrices - off;
        i2 = i1 + dim * n_matrices;
        i3 = K + 2 < I ? K + 2 : I;
        for (int j = I1; j <= i3; ++j)
        {
            h1r = rV[i1];
            h1i = iV[i1];
            h2r = rV[i2];
            h2i = iV[i2];
            zr = h1r;
            zi = h1i;
            cmul(zr, zi, t1r, t1i);
            cadd(zr, zi, h2r * t2, h2i * t2);
            csub(h1r, h1i, zr, zi);
            cmulc(zr, zi, v1r, v1i);
            csub(h2r, h2i, zr, zi);
            rV[i1] = h1r;
            iV[i1] = h1i;
            rV[i2] = h2r;
            iV[i2] = h2i;
            i1 += n_matrices;
            i2 += n_matrices;
        }
    }

    i1 = (I + (I - 1) * dim) * n_matrices - off;
    zi = iV[i1];
    if (zi != 0.0)
    {
        zr = rV[i1];
        d = cabs(zr, zi);
        cdiv(zr, zi, d);
        zi = -zi;
        rV[i1] = d;
        iV[i1] = 0.0;
        scale_rc(rV, iV, zr, zi, I, I1, I2, dim, n_matrices);
    }

}

#ifdef LAPACK

extern "C" void zgeev_( char* jobvl, char* jobvr, int* n, double* a,
        int* lda, double* w, double* vl, int* ldvl, double* vr, int* ldvr,
        double* work, int* lwork, double* rwork, int* info );

extern "C" void zlahqr_(int* WANTT, int* WANTZ, int* N, int* ILO, int* IHI, double* H, int* LDH,
			double* W, int* ILOZ, int* IHIZ, double* Z, int* LDZ, int* INFO);

void zlahqr__(double* rC, double* iC, double* rE, double* iE, double* rV, double* iV, int dim, int block_cnt, int thread_cnt)
{
    int n_matrices = thread_cnt;
    double *H = new double[2 * dim * dim];
    double *W = new double[2 * dim];
    double *Z = 0;

    for (int i = 0; i < dim * dim - dim; i++)
    {
        H[2*i+0] = 0.0;
        H[2*i+1] = 0.0;
    }

    double cr = rC[dim * n_matrices];
    double ci = iC[dim * n_matrices];

    int j = (dim * dim - dim);
    for (int i = 0; i < dim; i++)
    {
        double a = rC[i * n_matrices];
        double b = iC[i * n_matrices];
        cdiv(a, b, cr, ci);
        H[2*(j + i)+0] = -a;
        H[2*(j + i)+1] = -b;
    }

    j = 1;
    for (int i = 0; i < dim - 1; i++)
    {
        H[2*(j + i * (1 + dim))+0] = 1.0;
    }

#ifdef ZGEEV
    int N = dim;
    int LDH = dim, LDZ = dim;
    int INFO = 0;

    int lwork;
    char c = 'N';
    ComplexDouble wkopt;
    double* work;
    double *rwork = new double[2 * dim];

    lwork = -1;
    zgeev_(&c, &c, &N, H, &LDH, W, Z, &LDZ, Z, &LDZ, (double*)&wkopt, &lwork, rwork, &INFO );
    lwork = (int)wkopt.r;
    work = new double[2 * lwork];

    zgeev_(&c, &c, &N, H, &LDH, W, Z, &LDZ, Z, &LDZ, work, &lwork, rwork, &INFO );

#else //ZGEEV
    int WANTT = 0, WANTZ = 0;
    int N = dim;
    int ILO = 1, ILOZ = 1;
    int IHI = dim, IHIZ = dim;
    int LDH = dim, LDZ = dim;
    int INFO = 0;

    zlahqr_(&WANTT, &WANTZ, &N, &ILO, &IHI, H, &LDH, W, &ILOZ, &IHIZ, Z, &LDZ, &INFO);

#endif //ZGEEV

    for (int i = 0; i < dim; i++)
    {
        rE[i * n_matrices] = W[2*i+0];
        iE[i * n_matrices] = W[2*i+1];
    }

    delete [] H;
    delete [] W;
}
#endif //LAPACK


cuda void create_companion_matrices(double* rC, double *iC, double *rM, double *iM, int dim, int thread_cnt)
{
  // creates companion matrices from the coefficients of the
  // polynomials
  // rC, iC: store the coefficients of the polynomials
  // rM, iM: contains at exit the companion matrices
  // dim: dimension of the matrices
  

  
  int n_matrices = thread_cnt;

  // geht in dieser Datenstruktur nicht ohne weiteres mit memset()
  for(int i=0; i < dim*dim - dim; i++)
    {
      rM[i*n_matrices] = 0.0;
      iM[i*n_matrices] = 0.0;
    }
  double cr = rC[dim*n_matrices];
  double ci = iC[dim*n_matrices];

  int j = (dim*dim - dim)*n_matrices;

  for(int i=0; i<dim; ++i)
    {
      double a = rC[i*n_matrices];
      double b = iC[i*n_matrices];
      cdiv(a, b, cr, ci);	// Normalisierung der Koeff.; größter ist 1
      rM[j + i*n_matrices] = -a;
      iM[j + i*n_matrices] = -b;
    }
  j = n_matrices;
  for(int i=0; i<dim-1; ++i)
    {
      rM[j + i*(n_matrices + dim*n_matrices)] = 1.0; // rechte Spalte sind nur einser
    }
}

cuda void set_diagonal(double *rA, double *iA, double r, double i, int dim, int block_cnt, int thread_cnt)
{
  // sets the diagonal of A to the complex value (r,i)

  int n_matrices = thread_cnt;
  int off = n_matrices + dim * n_matrices;

  int ind = (1 + 1 * dim) * n_matrices - off;
  for(int j = 1; j <= dim; ++j)
    {
      rA[ind] = r;
      iA[ind] = i;
      ind += n_matrices + dim * n_matrices;
    }
}

cuda void make_identity(double* rI, double* iI, int dim, int block_cnt, int thread_cnt)
{
  // creates identity matrices in the arrays rI, iI
  // rI and iI are overwritten!!

  int n_matrices = thread_cnt;
  int off = n_matrices + dim * n_matrices;
  
  int I;
  for(int i=1; i <= dim; ++i)
    {
      I = (i + 1 * dim) * n_matrices - off;
      for(int j=1; j<=dim; ++j)
	{
	  rI[I] = 0.0;
	  iI[I] = 0.0;
	  I += dim * n_matrices;
	}
    }

  I = (1 + 1 * dim) * n_matrices - off;
  for(int i=1; i <= dim; ++i)
    {
      //      int j = (i + i * dim) * n_matrices - off;
      rI[I] = 1.0;
      I += (1 + dim) * n_matrices;
    }
}

cuda void zlahqr_schur(bool wantT, bool wantZ, int dim, double* rV, double* iV,
		       double* rE, double* iE, double* rZ, double* iZ, int block_cnt, int thread_cnt)
{
  // calculates eigenvalues and schur decomposition of upper hessenberg matrix rV, iV
  // wantT  true:  schur matrix T required
  //        false: T not required
  // wantZ  true:  matrix of schur vectors required
  //        false: not required
  //

  
    int n_matrices = thread_cnt;

    int K, L, M, I1, I2 = 0;
    int ILO = 1, IHI = dim;
    int i1, i2, i3, i4, i5, i6;
    double tr, ti;
    double v0r = 0.0, v0i = 0.0, v1r = 0.0, v1i = 0.0;
    double d, wr, sx;
    double wr1, wi1, wr2, wi2, wr3, wr4, wi4, wr5, wi5, wr6, zr, zi;
    double tst, d2, d3, aa, ab, ba, bb, s;
    double h1r, h1i, h2r, h2i;

    int ILOZ = 1, IHIZ = dim;	// alwas assume to work on the whole matrix
    
    int off = n_matrices + dim * n_matrices, off2 = n_matrices;

    double safemin = DBL_MIN;
    double ulp = DBL_EPSILON;
    double smlnum = safemin * (((double) (IHI - ILO + 1)) / ulp);

    // omitted:
    // clear trash (Loop 10)
    // ensure subdiagonals are real (start at line 270, loop 20)
    //
    //

    int JLO = ILO, JHI = IHI;
    if(wantT)
      {
	JLO = 1;
	JHI = dim;
      }

    // loop 20, ensure subdiagonals are real
    for(int i = ILO+1; i <= IHI; ++i)
      {
	i1 = (i + (i - 1) * dim) * n_matrices - off;
	v0i = iV[i1];
	if(v0i != 0.0)
	  {
	    v0r = rV[i1];
	    d = cabs1(v0r, v0i);
	    wr1 = v0r;
	    wi1 = v0i;
	    cdiv(wr1, wi1, d);
	    d = cabs(wr1, wi1);
	    wi1 = -wi1;
	    cdiv(wr1, wi1, d);
	    d = cabs(v0r, v0i);
	    rV[i1] = d;
	    iV[i1] = 0.0;

	    // sc from Lapack is here (w1r, w1i)
	    i2 = (i + i * dim) * n_matrices - off;
	    for(int j = 0; j < JHI - i + 1; ++j)
	      {
		/*
		h1r = rV[i2];
		h1i = iV[i2];
		cmul(h1r, h1i, wr1, wi1);
		*/
		cmul(rV[i2], iV[i2], wr1, wi1);
		i2 += dim * n_matrices;
	      }

	    i2 = (JLO + i *dim) * n_matrices - off;
	    for(int j = 0; j < min(JHI, i+1) - JLO + 1; ++j)
	      {
		/*
		h1r = rV[i2];
		h1i = iV[i2];
		cmulc(h1r, h1i, wr1, wi1);
		*/
		cmulc(rV[i2], iV[i2], wr1, wi1);
		i2 += n_matrices;
	      }

	    if(wantZ)
	      {
		i2 = (ILOZ + i * dim) * n_matrices - off;
		for(int j = 0; j < IHIZ - ILOZ + 1; ++j)
		  {
		    /*
		    v0r = rZ[i2];
		    v0i = iZ[i2];
		    cmulc(v0r, v0i, wr1, wi1);
		    */
		    cmulc(rZ[i2], iZ[i2], wr1, wi1);
		    i2 += n_matrices;
		  }
	      }
	  }
      }

		      
    
    if(wantT)			// line 309
      {
	I1 = 1;
	I2 = dim;
      }
    
    for (int I = IHI; I >= ILO; --I) // line 325, loop 30
    {
        L = ILO;
	// itmax not set, assumed as 30
	// 
        for (int ITS = 0; ITS <= 30; ++ITS) // loop 130
        {
            for (K = I; K >= L + 1; --K)
            {
                i4 = (K + (K - 1) * dim) * n_matrices - off;
                wr4 = rV[i4];
                wi4 = iV[i4];

                if (cabs1(wr4, wi4) <= smlnum)
                {
                    break;
                }

                i1 = (K - 1 + (K - 1) * dim) * n_matrices - off;
                i2 = (K + K * dim) * n_matrices - off;
                wr1 = rV[i1];
                wi1 = iV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                tst = cabs1(wr1, wi1) + cabs1(wr2, wi2);

                if (tst == 0.0)
                {
                    if (K - 2 >= ILO)
                    {
                        i3 = (K - 1 + (K - 2) * dim) * n_matrices - off;
                        wr3 = rV[i3];
                        tst += dabs(wr3);
                    }
                    if (K + 1 <= IHI)
                    {
                        i6 = (K + 1 + K * dim) * n_matrices - off;
                        wr6 = iV[i6];
                        tst += dabs(wr6);
                    }
                }

                if (dabs(wr4) <= ulp * tst)
                {
                    i5 = (K - 1 + K * dim) * n_matrices - off;
                    wr5 = rV[i5];
                    wi5 = iV[i5];
                    d2 = cabs1(wr4, wi4);
                    d3 = cabs1(wr5, wi5);
                    ab = fmax(d2, d3);
                    ba = fmin(d2, d3);
                    zr = wr1 - wr2;
                    zi = wi1 - wi2;
                    d2 = cabs1(wr2, wi2);
                    d3 = cabs1(zr, zi);
                    aa = fmax(d2, d3);
                    bb = fmin(d2, d3);
                    s = aa + ab;

                    if (ba * (ab / s) <= fmax(smlnum, ulp * (bb * (aa / s))))
                    {
                        break;
                    }
                }
            }

            L = K;

            if (L > ILO)
            {
	      // H(L,L-1) is negligible
	      // H(L,L-1) is (rV[i1], iV[i1]) as complex number with i1 as below
                i1 = (L + (L - 1) * dim) * n_matrices - off;
                rV[i1] = 0.0;
                iV[i1] = 0.0;
            }

            if (L >= I)
            {
	      // if submatrix of order one has split off
                break;
            }

	    // Active submatrix is in rows and columns L to I.
	    // If eigenvalues only are computed, only the
	    // active submatrix need to be transformed
	    if(!wantT)
	      {
		I1 = L;
		I2 = I;
	      }

	    
            if (ITS == 10)
            {
	      // exceptional shift
                i1 = (L + 1 + L * dim) * n_matrices - off;
                i2 = i1 - n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75; // dat1 == 0.75
                tr = d + wr2;
                ti = wi2;
            }
            else if (ITS == 20)
            {
	      // exceptional shift
                i1 = (I + (I - 1) * dim) * n_matrices - off;
                i2 = i1 + dim * n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75; // dat1 == 0.75
                tr = d + wr2;
                ti = wi2;
            }
            else
            {
	      // Wilkinson's shift
	      
                double ur, ui, vr, vi, h3r, h3i, xr, xi, yr, yi, z1r, z1i, z2r, z2i;

                i1 = (I + I * dim) * n_matrices - off;
                i2 = (I - 1 + I * dim) * n_matrices - off;
                i3 = (I + (I - 1) * dim) * n_matrices - off;
                i4 = ((I - 1) + (I - 1) * dim) * n_matrices - off;

                tr = rV[i1];
                ti = iV[i1];
                h2r = rV[i2];
                h2i = iV[i2];
                h3r = rV[i3];
                h3i = iV[i3];

                csqrt(h2r, h2i);
                csqrt(h3r, h3i);
                cmul(h2r, h2i, h3r, h3i);
                ur = h2r;
                ui = h2i;
                s = cabs1(ur, ui);
                if (s != 0.0)
                {
                    xr = rV[i4];
                    xi = iV[i4];
                    csub(xr, xi, tr, ti);
                    cmul(xr, xi, 0.5);
                    sx = cabs1(xr, xi);
                    s = fmax(s, sx);
                    z1r = xr;
                    z1i = xi;
                    cdiv(z1r, z1i, s);
                    z2r = ur;
                    z2i = ui;
                    cdiv(z2r, z2i, s);
                    cmul(z1r, z1i, z1r, z1i);
                    cmul(z2r, z2i, z2r, z2i);
                    cadd(z1r, z1i, z2r, z2i);
                    csqrt(z1r, z1i);
                    cmul(z1r, z1i, s);
                    yr = z1r;
                    yi = z1i;

                    if (sx > 0.0)
                    {
                        zr = xr / sx;
                        zi = xi / sx;

                        if (zr * yr + zi * yi < 0.0)
                        {
                            yr = -yr;
                            yi = -yi;
                        }
                    }
                    vr = ur;
                    vi = ui;
                    cadd(xr, xi, yr, yi);

#ifdef MKL
                    cdivex(ur, ui, xr, xi, ur, ui);
#else
                    cdiv(ur, ui, xr, xi);
#endif

                    cmul(ur, ui, vr, vi);
                    csub(tr, ti, ur, ui);
                }
            } // Wilkinson's shift end
	    // shift calculated

	    // Loop 60 in Line 421 omitted because this decreases
	    // performance in vectorized execution
	    // this loop saves some computations in special cases
	    
            double d1r;
            M = L;

            i1 = (L + L * dim) * n_matrices - off;
            i3 = (L + 1 + L * dim) * n_matrices - off;

            h1r = rV[i1];
            h1i = iV[i1];
            d1r = rV[i3];

            zr = h1r - tr;
            zi = h1i - ti;
            s = cabs1(zr, zi) + dabs(d1r);
            cdiv(zr, zi, s);
            d1r /= s;
            v0r = zr;
            v0i = zi;
            v1r = d1r;
            v1i = 0.0;

             single_shift_qr_schur(wantT, wantZ, rV, iV, rZ, iZ, I, L, M, I1, I2, v0r, v0i, v1r, v1i, dim, n_matrices);

        }
    }

    for (int I = IHI; I >= ILO; --I)
    {
        i1 = (I + I * dim) * n_matrices - off;
        i2 = I * n_matrices - off2;
        zr = rV[i1];
        zi = iV[i1];
        rE[i2] = zr;
        iE[i2] = zi;
    }
}

cuda void zgemv_simple(bool hermitian, int m, int n, double *rA, double *iA, int dim, double *rx, double *ix,
		       double rbeta, double ibeta, double *ry, double *iy, int block_cnt, int thread_cnt)
{
  // computes A*x + beta*y
  // A is seen as a m×n matrix, in the original data structure A has dimension dim×dim
  // y is assumed to be a column of a dim×dim matrix, it is the pointer to the first element in the column
  // x and have a suitable dimension
  // output is in y
  // if hermitian is true, then A**H*x + beta*y is computed

  int n_matrices = thread_cnt;
  int offm = n_matrices + dim * n_matrices;
  int offv = n_matrices;

  int indm;			// matrix index
  int ind1;			// vector index 1
  int ind2;			// vector index 2

  double rv, iv;		// real/imaginary vector entry
  double rm, im;		// real/imaginary matrix entry
  double rtemp, itemp;

  int lenx, leny;
  if(hermitian)
    {
      lenx = m;
      leny = n;
    }
  else
    {
      lenx = n;
      leny = m;
    }

  // Set y := beta*y

  // omitted test for beta = 1+0i
  ind1 = 1 * n_matrices - offv;
  for(int i = 1; i <= leny; ++i)
    {
      //ind1 = i * n_matrices - offv;
      rv = ry[ind1];
      iv = iy[ind1];
      cmul(rv, iv, rbeta, ibeta);
      ry[ind1] = rv;
      iy[ind1] = iv;
      ind1 += n_matrices;
    }

  if(!hermitian)
    {
      // Set y := A*x + y

      for(int j=1; j<=n; ++j)
	{
	  ind1 = j * n_matrices - offv;
	  rtemp = rx[ind1];
	  itemp = ix[ind1];
	  for(int i=1; i<=m; ++i)
	    {
	      ind2 = i * n_matrices - offv;
	      indm = (i + j * dim) * n_matrices - offm;
	      rv = ry[ind2];
	      iv = iy[ind2];
	      rm = rA[indm];
	      im = iA[indm];
	      cmul(rm, im, rtemp, itemp);
	      cadd(rv, iv, rm, im);
	      ry[ind2] = rv;
	      iy[ind2] = iv;
	    }
	}
    }
  else				// hermitian is true
    {
      for(int j=1; j<=n; ++j)
	{
	  rtemp = 0.0;
	  itemp = 0.0;
	  for(int i=1; i<=m; ++i)
	    {
	      ind2 = i * n_matrices - offv;
	      indm = (i + j * dim) * n_matrices - offm;
	      rm = rA[indm];
	      im = iA[indm];
	      rv = rx[ind2];
	      iv = ix[ind2];
	      cmulc(rv, iv, rm, im);
	      cadd(rtemp, itemp, rv, iv);
	    }
	  ind1 = j * n_matrices - offv;
	  rv = ry[ind1];
	  iv = ry[ind1];
	  cadd(rv, iv, rtemp, itemp);
	  ry[ind1] = rv;
	  iy[ind1] = iv;
	}
    }
}
	
cuda void ztrevc(double* rT, double* iT, double* rL, double* iL, double* rR, double* iR,
		 double *rwork, double *iwork, int dim, int block_cnt, int thread_cnt)
{
  // computes all eigenvectors of triangular matrix rT, iT (real, imaginary part)
  // R, L contain on entry a square matrix, usually the unitary matrix Q of schur vectors
  //   on exit they contain the right, left eigenvectors multiplied from left with Q
  // T stays unchanged
  // rwork, iwork are work arrays, they must have length at least 2*dim*thread_cnt
  //   the diagonal and also the right-hand side for the backsubstitution are stored

  // LAPACK arguments Side and HOWMNY are not yet implemented
  // LAPACK-switch over is assumed to be true, depends on Side and HowMny
  // Side assumed to be both, that is left and right eigenvectors
  // HowMny assumed to be B, this means all eigenvectors are backtransformed
  //   by R and L, respectively
  
  int n_matrices = thread_cnt;
  int off = n_matrices + dim * n_matrices; // offset for matrix-arrays
  int off2 = n_matrices;		   // offset for vector-arrays
  
  double unfl = DBL_MIN;
  double ulp = DBL_EPSILON;
  double smlnum = unfl * ( (double)dim / ulp);

  int i1, i2;
  double zr, zi;
  double t1r, t1i;
  
  double *rdiag = rwork;
  double *idiag = iwork;
  
  // store diagonal of T in work array
  for(int i=1; i<=dim; ++i)
    {
      i1 = (i + i * dim) * n_matrices - off;
      i2 = i * n_matrices - off2;
      zr = rT[i1];
      zi = iT[i1];
      rdiag[i2] = zr;
      idiag[i2] = zi;
    }

  
  // compute 1-norm of each column of the strictly upper triangular part of T
  // to control overflow in triangular solver
  /*
    omitted, as this is only needed for the handling of ill conditined problems
    this functionality is not implemented
  double *work2 = new double[dim*n_matrices];
  work2[0] = 0.0;
  for(int i=2; i<=dim; ++i)	// Loop 30
    {
      double norm = 0.0;
      for(int j=1; j<=i-1; ++j)
	{
	  i1 = (j + i * dim) * n_matrices - off;
	  t1r = rT[i1];
	  t1i = iT[i1];
	  norm += cabs1(t1r, t1i);
	}
      i2 = i * n_matrices - off2;
      work2[i2] = norm;
    }
  */

  
  // if(rightv) assumed to be always true
  int is = dim;
  double *rrhs = rwork + dim * n_matrices;
  double *irhs = iwork + dim * n_matrices;
    
  for(int ki = dim; ki>=1; --ki) // Loop 80
    {
      // selection of special eigenvectors not implemented
      i1 = (ki + ki * dim) * n_matrices - off;
      t1r = rT[i1];
      t1i = iT[i1];
      double smin = cabs1(t1r, t1i);
      smin *= ulp;
      smin = fmax(smin, smlnum);
      
      // form right-hand side

      i1 = (1 + ki * dim) * n_matrices - off;
      i2 = 1 * n_matrices - off2;
      for(int k=1; k<=ki-1; ++k) // Loop 40
	{
	  //i1 = (k + ki * dim) * n_matrices - off; // maybe adjust the calculation
	  //i2 = k * n_matrices - off2;
	  zr = rT[i1];
	  zi = iT[i1];
	  rrhs[i2] = -zr;
	  irhs[i2] = -zi;
	  i1 += n_matrices;
	  i2 += n_matrices;
	}

      // solve triangular system
      // (T(1:ki-1,1:ki-1)-T(ki,ki)*Id(ki-1))*x = scale*work
      

      // set T(k,k) = T(k,k)-T(ki,ki)
      i1 = (1 + 1 * dim) * n_matrices - off;
      i2 = (ki + ki * dim) * n_matrices - off;
      for(int k=1; k<=ki-1; ++k) // Loop 50
	{
	  //i1 = (k + k * dim) * n_matrices - off; // maybe the calculation can be done more efficiently
	  //i2 = (ki + ki * dim) * n_matrices - off;
	  zr = rT[i2];
	  zi = iT[i2];
	  t1r = rT[i1];
	  t1i = iT[i1];
	  csub(t1r, t1i, zr, zi);
	  
	  // can lead to thread divergence!!!!!
	  // possibly let it out
	  if(cabs1(t1r,t1i)<smin)
	    {
	      t1r = smin;
	      t1i = 0.0;
	    }
	  rT[i1] = t1r;
	  iT[i1] = t1i;

	  i1 += (1 + 1 * dim) * n_matrices;
	} // Loop 50
      
      if(ki > 1)		// should not lead to thread divergence as all threads execute it or not
	{
	  backsubstitution(ki-1, rT, iT, rrhs, irhs, dim, n_matrices);
	  /* the case over=false is not implemented
	}

      // copy vector x or Q*x to rR, iR

      if(ki > 1)
	{
	  */
	  // zgemv('N', n, ki-1, 1+0i, R, n, work(1), 1,
	  i1 = (1 + ki * dim) * n_matrices - off;

	  zgemv_simple(false, dim, ki-1, rR, iR, dim, rrhs, irhs, 1.0, 0.0,
		       rR + i1, iR + i1, block_cnt, thread_cnt);
	  // scaling of the eigenvector to fulfill the requirements, i.e. beeing not large
	  // in this code it is normalized with respect to the max-norm
	  i1 = (1 + ki * dim) * n_matrices - off;
	  double max = 0.0;
	  double temp;
	  for(int k = 1; k <= dim; ++k)
	    {
	      t1r = rT[i1];
	      t1i = iT[i1];
	      temp = cabs1(t1r, t1i);
	      max = temp>max ? temp : max;
	      i1 += n_matrices;
	    }
	  temp = 1.0 / max;
	  i1 = (1 + ki * dim) * n_matrices - off;
	  for(int k = 1; k <= dim; ++k)
	    {
	      cmul(rT[i1], iT[i1], temp);
	      i1 += n_matrices;
	    }
	}
	  

      // setting the diagonal elements
      i1 = (1 + 1 * dim) * n_matrices - off;
      i2 = 1 * n_matrices - off2;
      for(int k = 1; k <= ki - 1; ++k)
	{
	  rT[i1] = rdiag[i2];
	  iT[i1] = idiag[i2];
	  i1 += (1 + 1 * dim) * n_matrices;
	  i2 += n_matrices;
	}
      
      is -= 1;
    } // big for loop (loop 80 in LAPACK)

}

cuda void zgehrd(double *rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
		 int dim, int block_cnt, int thread_cnt)
{
  // reduces the matrix A to upper Hessenberg form by a unitary similarity transform
  // the transform is represented as product of elementary Householder reflectors H=I-tau*v*v**H
  // the values of tau are stored in the corresponding array
  // the values of v are stored in the now unused part of A, the first value is always 1.0

  int n_matrices = thread_cnt;
  int ILO = 1;
  int IHI = dim;

  double ralpha, ialpha;
  double a0r, a0i, t0r, t0i, temp;
  double norm, beta;
  double rtemp, itemp;

  double safemin = DBL_MIN;
  double rsafemin = 1.0 / safemin;

  int n;
  int off = dim * n_matrices + n_matrices;
  int offv = n_matrices;
  int ind, ind2, ind3, ind4;
  
  // omitted setting tau to 0.0 for ranges outside of ILO to IHI-1

  // directly use unblocked code, zgehd2
  for(int i = ILO; i <= IHI - 1; ++i)
    {
      ind = (i + 1 + i*dim)*n_matrices - off;
      ralpha = rA[ind];
      ialpha = iA[ind];

      // zlarfg directly coded here
      // simplify the code dramatically and don't compute address specil cases
      // that is assume tere actually is a reflector to compte (norm>0) and absolute value of beta is large enough

      int n = IHI - i;
      norm = 0.0;
      ind2 = min(i+2,dim);
      ind2 = (ind2 + i*dim)*n_matrices - off;
      for(int j = 1; j <= n - 1; ++j)
	{
	  a0r = rA[ind2];
	  a0i = iA[ind2];
	  norm += a0r*a0r + a0i*a0i;
	  ind2 += n_matrices;
	}
      //norm = sqrt(norm); not needed here and later on squared again

      // omitted check for norm==0, makes a early return possible
      temp = sqrt(ralpha*ralpha + ialpha*ialpha + norm);
      beta = copysign( temp, ralpha);
      temp = 1.0 / beta;
      t0r = (beta - ralpha) * temp;
      t0i = -ialpha * temp;
      ralpha -= beta;
      cinv(ralpha, ialpha);
      ind2 = min(i+2, dim);
      ind2 = (ind2 + i * dim) * n_matrices - off;
      for(int j = 1; j <= n - 1; ++j)
	{
	  a0r = rA[ind2];
	  a0i = iA[ind2];
	  cmul(a0r, a0i, ralpha, ialpha);
	  rA[ind2] = a0r;
	  iA[ind2] = a0i;
	  ind2 += n_matrices;
	}
      ralpha = beta;
      ialpha = 0.0;

      // set tau
      ind2 = i * n_matrices - offv;
      rtau[ind2] = t0r;
      itau[ind2] = t0i;
      
      // end of zlarfg
      
      ind = (i + 1 + i * dim) * n_matrices - off; //should be still the value from above
      rA[ind] = 1.0;
      iA[ind] = 0.0;

      // apply H from the right
      // incv is alwas 1 in the following (also for subroutines)
      // 
      // call to zlarf with parameters
      // 'Right', ihi, ihi-i, a(i+1,i), 1, tau(i), a(1, i+1), lda, work
      // 'Right' means applyleft is false
      int lastv = IHI - i;
      int lastc = IHI;

      ind2 = (1 + (i + 1) * dim) * n_matrices - off;
      double *rC = rA + ind2;
      double *iC = iA + ind2;

      double *rv = rA + ind;
      double *iv = iA + ind;

      zgemv_simple(false, lastc, lastv, rC, iC, dim, rv, iv,
		   0.0, 0.0, rwork, iwork, block_cnt, thread_cnt);
      // now call to  zgerc
      // alpha is rtau(i), itau(i)
      // m = lastc, n = lastv
      ind2 = i * n_matrices - offv;
      ind3 = 1 * n_matrices - offv; // used for the index of v
      for(int j = 1; j <= lastv; ++j)
	{
	  rtemp = -rtau[ind2];
	  itemp = -itau[ind2];
	  cmulc(rtemp, itemp, rv[ind3], iv[ind3]);
	  ind4 = n_matrices - offv;
	  for(int k = 1; k <= lastc; ++k)
	    {
	      ind = (k + j * dim) * n_matrices - off;
	      a0r = rwork[ind4];
	      a0i = iwork[ind4];
	      cmul(a0r, a0i, rtemp, itemp);
	      cadd(a0r, a0i, rC[ind], iC[ind]);
	      rC[ind] = a0r;
	      iC[ind] = a0i;
	      ind4 += n_matrices;
	    }
	  ind3 += n_matrices;
	}
      // end of zgerc
      
      // apply H from the left

      // call to zlarf with parameters
      // 'Left', ihi-i, n-i, a(i+1,i), 1, dconjg(tau(i)), a(i+1, i+1), lda, work

      lastv = IHI - i;
      lastc = dim - i;

      ind2 = (i + 1 + (i + 1) * dim) * n_matrices - off;
      rC = rA + ind2;
      iC = iA + ind2;
      ind = (i + 1 + i * dim) * n_matrices - off;
      rv = rA + ind;
      iv = iA + ind;
      zgemv_simple(true, lastv, lastc, rC, iC, dim, rv, iv,
		   0.0, 0.0, rwork, iwork, block_cnt, thread_cnt);
      // call to zgerc

      ind2 = i * n_matrices - offv;
      ind3 = 1 * n_matrices - offv;
      for(int j = 1; j <= lastc; ++j)
	{
	  rtemp = -rtau[ind2];
	  itemp = itau[ind2];	// the sign is due to the call to zlarf with dconjg(tau(i)), changing the sign
	  cmulc(rtemp, itemp, rwork[ind3], iwork[ind3]);
	  for(int k = 1; k <= lastv; ++k)
	    {
	      ind = (k + j * dim) * n_matrices - off;
	      ind4 = k * n_matrices - offv;
	      a0r = rv[ind4];
	      a0i = iv[ind4];
	      cmul(a0r, a0i, rtemp, itemp);
	      cadd(a0r, a0i, rC[ind], iC[ind]);
	      rC[ind] = a0r;
	      iC[ind] = a0i;
	    }
	  ind3 += n_matrices;
	}
      // end of zgerc

      // end of both calls to zlarf

      ind = (i + 1 + i * dim) * n_matrices - off;
      rA[ind] = ralpha;
      iA[ind] = ialpha;
    }
    
}

cuda void zunghr(double *rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
		 int dim, int block_cnt, int thread_cnt)
{
  // A is a matrix containing the vectors for the elementary reflectors as returned by zgehrd
  //   on exit A is overwritten with the unitary matrix Q defined by the product
  //   of the elementary reflectors
  // tau is the vector for the scalar factors of the reflectors, as returned by zgehrd
  // 
  int n_matrices = thread_cnt;
  int offm = n_matrices + dim * n_matrices;
  int offv = n_matrices;
  int IHI = dim;
  int ILO = 1;
  int nh = IHI - ILO;

  int ind, ind2, ind3, ind4;
  int lastv, lastc;

  double *rA_temp, *iA_temp;
  rA_temp = rA;
  iA_temp = iA;
  double *rC, *iC;
  double *rv, *iv;
  double rtemp, itemp;
  double a0r, a0i;
  

  // the vectors defining the reflectors are shifted one column to the right.
  //
  // the creation of a unit matrix in the rows and colimns before ILO and after IHI
  //   omitted, as ILO and IHI are set such that the whole matrix is considered

  for(int j = IHI; j >= ILO + 1; --j)
    {
      ind = (1 + j * dim) * n_matrices - offm;
      for(int i = 1; i <= j - 1; ++i)
	{
	  rA[ind] = 0.0;
	  iA[ind] = 0.0;
	  ind += n_matrices;
	}

      ind = (j+1 + j * dim) * n_matrices - offm;
      for(int i = j + 1; i <= IHI; ++i)
	{
	  //ind = (i + j * dim) * n_matrices - offm;
	  ind2 = ind - dim * n_matrices;
	  rA[ind] = rA[ind2];
	  iA[ind] = iA[ind2];
	  ind += n_matrices;
	}
      // third LAPACK loop (30) not needed
    }

  for(int j = 1; j <= ILO; ++j)
    {
      ind = (1 + j * dim) * n_matrices - offm;
      for(int i = 1; i <= dim; ++i)
	{
	  rA[ind] = 0.0;
	  iA[ind] = 0.0;
	  ind += n_matrices;
	}
      ind2 = (j + j * dim) * thread_cnt - offm;
      rA[ind2] = 1.0;
      iA[ind2] = 0.0;
    }
  
  // LAPACK loop 80 omitted

  // call to zungqr, then use unblocked code, i.e. zung2r
  ind = (ILO+1 + (ILO+1) * dim) * thread_cnt - offm;
  rA = rA + ind;
  iA = iA + ind;
  // zung2r
  // first loop 20 omitted as this does nothing if IHI = dim

  // loop 40
  for(int i = nh; i >= 1; --i)
    {
      // apply H(i) from the left to A(i:nh, i:nh)
      // the bounds are due to the call in zunghr

      // if check omitted, always evaluates to true
      
      ind = (i + i * dim) * n_matrices - offm;
      rA[ind] = 1.0;
      iA[ind] = 0.0;

      // call to zlarf
      // 'Left', nh-i+1, nh-i, a(i,i), 1, tau(i), a(i, i+1), lda, work

      lastv = nh - i +1;
      lastc = nh - i;

      ind2 = (i + (i + 1) * dim) * n_matrices - offm;
      rC = rA + ind2;
      iC = iA + ind2;
      ind = (i + i * dim) * n_matrices - offm;
      rv = rA + ind;
      iv = iA + ind;
      zgemv_simple(true, lastv, lastc, rC, iC, dim, rv, iv,
		   0.0, 0.0, rwork, iwork, block_cnt, thread_cnt);
      // call to zgerc

      ind2 = i * n_matrices - offv;
      ind3 = 1 * n_matrices - offv;
      for(int j = 1; j <= lastc; ++j)
	{
	  rtemp = -rtau[ind2];
	  itemp = -itau[ind2];
	  cmulc(rtemp, itemp, rwork[ind3], iwork[ind3]);
	  for(int k = 1; k <= lastv; ++k)
	    {
	      ind = (k + j * dim) * n_matrices - offm;
	      ind4 = k * n_matrices - offv;
	      a0r = rv[ind4];
	      a0i = iv[ind4];
	      cmul(a0r, a0i, rtemp, itemp);
	      cadd(a0r, a0i, rC[ind], iC[ind]);
	      rC[ind] = a0r;
	      iC[ind] = a0i;
	    }
	  ind3 += n_matrices;
	}
      // zgerc finished

      // return from zlarf

      // check for i<dim omitted, always true
      // scale values in column i
      ind = i * n_matrices - offv;
      rtemp = -rtau[ind];
      itemp = -itau[ind];
      ind = (i+1 + i * dim) * n_matrices - offm;
      for(int j = 1; j <= nh-i; ++j)
	{
	  a0r = rA[ind];
	  a0i = iA[ind];
	  cmul(a0r, a0i, rtemp, itemp);
	  rA[ind] = a0r;
	  iA[ind] = a0i;

	  ind += n_matrices;
	}

      ind = (i + i * dim) * n_matrices - offm;
      ind2 = i * n_matrices - offv;
      rtemp = 1.0;
      itemp = 0.0;
      csub(rtemp, itemp, rtau[ind2], itau[ind2]);
      rA[ind] = rtemp;
      iA[ind] = itemp;

      ind = (1 + i * dim) * n_matrices - offm;
      for(int j = 1; j <= i-1; ++j)
	{
	  rA[ind] = 0.0;
	  iA[ind] = 0.0;
	  ind += n_matrices;
	}
      
    } // big for loop over columns

  rA = rA_temp;
  iA = iA_temp;
  
}


  
cuda void zlahqr_(double* rE, double* iE, double* rV, double* iV, int dim, int block_cnt, int thread_cnt)
{
    int n_matrices = thread_cnt;

    int K, L, M, I1, I2;
    int ILO = 1, IHI = dim;
    int i1, i2, i3, i4, i5, i6;
    double tr, ti;
    double v0r = 0.0, v0i = 0.0, v1r = 0.0, v1i = 0.0;
    double d, wr, sx;
    double wr1, wi1, wr2, wi2, wr3, wr4, wi4, wr5, wi5, wr6, zr, zi;
    double tst, d2, d3, aa, ab, ba, bb, s;
    double h1r, h1i, h2r, h2i;

    int off = n_matrices + dim * n_matrices, off2 = n_matrices;

    double safemin = DBL_MIN;
    double ulp = DBL_EPSILON;
    double smlnum = safemin * (((double) (IHI - ILO + 1)) / ulp);

    for (int I = IHI; I >= ILO; --I)
    {
        L = ILO;
        for (int ITS = 0; ITS <= 30; ++ITS)
        {
            for (K = I; K >= L + 1; --K)
            {
                i4 = (K + (K - 1) * dim) * n_matrices - off;
                wr4 = rV[i4];
                wi4 = iV[i4];

                if (cabs1(wr4, wi4) <= smlnum)
                {
                    break;
                }

                i1 = (K - 1 + (K - 1) * dim) * n_matrices - off;
                i2 = (K + K * dim) * n_matrices - off;
                wr1 = rV[i1];
                wi1 = iV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                tst = cabs1(wr1, wi1) + cabs1(wr2, wi2);

                if (tst == 0.0)
                {
                    if (K - 2 >= ILO)
                    {
                        i3 = (K - 1 + (K - 2) * dim) * n_matrices - off;
                        wr3 = rV[i3];
                        tst += dabs(wr3);
                    }
                    if (K + 1 <= IHI)
                    {
                        i6 = (K + 1 + K * dim) * n_matrices - off;
                        wr6 = iV[i6];
                        tst += dabs(wr6);
                    }
                }

                if (dabs(wr4) <= ulp * tst)
                {
                    i5 = (K - 1 + K * dim) * n_matrices - off;
                    wr5 = rV[i5];
                    wi5 = iV[i5];
                    d2 = cabs1(wr4, wi4);
                    d3 = cabs1(wr5, wi5);
                    ab = fmax(d2, d3);
                    ba = fmin(d2, d3);
                    zr = wr1 - wr2;
                    zi = wi1 - wi2;
                    d2 = cabs1(wr2, wi2);
                    d3 = cabs1(zr, zi);
                    aa = fmax(d2, d3);
                    bb = fmin(d2, d3);
                    s = aa + ab;

                    if (ba * (ab / s) <= fmax(smlnum, ulp * (bb * (aa / s))))
                    {
                        break;
                    }
                }
            }

            L = K;

            if (L > ILO)
            {
                i1 = (L + (L - 1) * dim) * n_matrices - off;
                rV[i1] = 0.0;
                iV[i1] = 0.0;
            }

            if (L >= I)
            {
                break;
            }

            I1 = L;
            I2 = I;

            if (ITS == 10)
            {
                i1 = (L + 1 + L * dim) * n_matrices - off;
                i2 = i1 - n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75;
                tr = d + wr2;
                ti = wi2;
            }
            else if (ITS == 20)
            {
                i1 = (I + (I - 1) * dim) * n_matrices - off;
                i2 = i1 + dim * n_matrices;
                wr = rV[i1];
                wr2 = rV[i2];
                wi2 = iV[i2];
                d = dabs(wr) * 0.75;
                tr = d + wr2;
                ti = wi2;
            }
            else
            {
                double ur, ui, vr, vi, h3r, h3i, xr, xi, yr, yi, z1r, z1i, z2r, z2i;

                i1 = (I + I * dim) * n_matrices - off;
                i2 = (I - 1 + I * dim) * n_matrices - off;
                i3 = (I + (I - 1) * dim) * n_matrices - off;
                i4 = ((I - 1) + (I - 1) * dim) * n_matrices - off;

                tr = rV[i1];
                ti = iV[i1];
                h2r = rV[i2];
                h2i = iV[i2];
                h3r = rV[i3];
                h3i = iV[i3];

                csqrt(h2r, h2i);
                csqrt(h3r, h3i);
                cmul(h2r, h2i, h3r, h3i);
                ur = h2r;
                ui = h2i;
                s = cabs1(ur, ui);
                if (s != 0.0)
                {
                    xr = rV[i4];
                    xi = iV[i4];
                    csub(xr, xi, tr, ti);
                    cmul(xr, xi, 0.5);
                    sx = cabs1(xr, xi);
                    s = fmax(s, sx);
                    z1r = xr;
                    z1i = xi;
                    cdiv(z1r, z1i, s);
                    z2r = ur;
                    z2i = ui;
                    cdiv(z2r, z2i, s);
                    cmul(z1r, z1i, z1r, z1i);
                    cmul(z2r, z2i, z2r, z2i);
                    cadd(z1r, z1i, z2r, z2i);
                    csqrt(z1r, z1i);
                    cmul(z1r, z1i, s);
                    yr = z1r;
                    yi = z1i;

                    if (sx > 0.0)
                    {
                        zr = xr / sx;
                        zi = xi / sx;

                        if (zr * yr + zi * yi < 0.0)
                        {
                            yr = -yr;
                            yi = -yi;
                        }
                    }
                    vr = ur;
                    vi = ui;
                    cadd(xr, xi, yr, yi);

#ifdef MKL
                    cdivex(ur, ui, xr, xi, ur, ui);
#else
                    cdiv(ur, ui, xr, xi);
#endif

                    cmul(ur, ui, vr, vi);
                    csub(tr, ti, ur, ui);
                }
            }

            double d1r;
            M = L;

            i1 = (L + L * dim) * n_matrices - off;
            i3 = (L + 1 + L * dim) * n_matrices - off;

            h1r = rV[i1];
            h1i = iV[i1];
            d1r = rV[i3];

            zr = h1r - tr;
            zi = h1i - ti;
            s = cabs1(zr, zi) + dabs(d1r);
            cdiv(zr, zi, s);
            d1r /= s;
            v0r = zr;
            v0i = zi;
            v1r = d1r;
            v1i = 0.0;

            single_shift_qr(rV, iV, I, L, M, I1, I2, v0r, v0i, v1r, v1i, dim, n_matrices);

        }
    }

    for (int I = IHI; I >= ILO; --I)
    {
        i1 = (I + I * dim) * n_matrices - off;
        i2 = I * n_matrices - off2;
        zr = rV[i1];
        zi = iV[i1];
        rE[i2] = zr;
        iE[i2] = zi;
    }
}

extern "C" double dlamch_(char *cmach);

global void zlahqr(double* rC, double* iC, double* rE, double* iE, double* rV, double* iV, double* rZ, double* iZ,
		   int dim, int dim1, int dim2, int block_cnt, int thread_cnt)
{

#ifndef CUDA

#ifdef MIC
    int n = block_cnt * thread_cnt * dim, n1 = block_cnt * thread_cnt * dim1, n2 = block_cnt * thread_cnt * dim2;
#pragma omp target map(alloc: rV[0:n2], iV[0:n2], rZ[0:n2], iZ[0:n2]) map(to:rC[0:n1], iC[0:n1]) map(from: rE[0:n], iE[0:n])
#endif //MIC

#ifdef OPENMP

#ifdef LAPACK
    char s;
    dlamch_(&s);
#endif //LAPACK
    
#pragma omp parallel for
#endif //OPENMP

#ifdef OPENACC
    int n = block_cnt * thread_cnt * dim, n1 = block_cnt * thread_cnt * dim1, n2 = block_cnt * thread_cnt * dim2;
#pragma acc parallel loop create(rV[0:n2],iV[0:n2],rZ[0:n2],iZ[0:n2]) copyin(rC[0:n1],iC[0:n1]) copyout(rE[0:n],iE[0:n])
#endif //OPENACC

    for (int block_idx = 0; block_idx < block_cnt; block_idx++)
    {
        for (int thread_idx = 0; thread_idx < thread_cnt; thread_idx++)
        {

#else // ndef CUDA

            int block_idx = blockIdx.x;
            int thread_idx = threadIdx.x;

#endif // ndef CUDA

	    create_companion_matrices(rC + thread_idx + block_idx * thread_cnt * dim1, iC + thread_idx + block_idx * thread_cnt * dim1,
				      rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2,
				      dim, thread_cnt);
	    /*   
	    ZLAHQR(rE + thread_idx + block_idx * thread_cnt * dim, iE + thread_idx + block_idx * thread_cnt * dim,
	           rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2,
		   dim, block_cnt, thread_cnt);
	    */
	    
	    make_identity(rZ + thread_idx + block_idx * thread_cnt * dim2,
			  iZ + thread_idx + block_idx * thread_cnt * dim2,
			  dim, block_cnt, thread_cnt);
	    
	    zlahqr_schur(true, true, dim,
			 rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2,
			 rE + thread_idx + block_idx * thread_cnt * dim, iE + thread_idx + block_idx * thread_cnt * dim,
			 rZ + thread_idx + block_idx * thread_cnt * dim2, iZ + thread_idx + block_idx * thread_cnt * dim2,
			 block_cnt, thread_cnt);
	    	      
#ifndef CUDA

        }
    }

#endif // ndef CUDA
}

// debugging purpose
global void test_ident(double* rZ, double* iZ, int dim, int block_cnt, int thread_cnt)
{
  int dim2 = dim * dim;
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for
#endif	// OPENMP
  for(int block_idx = 0;  block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA
	  make_identity(rZ + thread_idx + block_idx * thread_cnt * dim2,
			iZ + thread_idx + block_idx * thread_cnt * dim2,
			dim, block_cnt, thread_cnt);
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

global void zlahqr_new(double* rE, double* iE, double* rV, double* iV, double* rZ, double* iZ,
		       int dim, int dim2, int block_cnt, int thread_cnt)
{
#ifndef CUDA

#ifdef OPENMP
#pragma omp parallel for schedule(static, 8)
#endif	// OPENMP

  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA

	  /*
	  make_identity(rZ + thread_idx + block_idx * thread_cnt * dim2,
			iZ + thread_idx + block_idx * thread_cnt * dim2,
			dim, block_cnt, thread_cnt);
	  */

	  /*
	  set_diagonal(rZ + thread_idx + block_idx * thread_cnt * dim2,
		       iZ + thread_idx + block_idx * thread_cnt * dim2,
		       1.0, 0.0, dim, block_cnt, thread_cnt);
	  */	  

	  zlahqr_schur(true, true, dim,
		       rV + thread_idx + block_idx * thread_cnt * dim2,
		       iV + thread_idx + block_idx * thread_cnt * dim2,
		       rE + thread_idx + block_idx * thread_cnt * dim,
		       iE + thread_idx + block_idx * thread_cnt * dim,
		       rZ + thread_idx + block_idx * thread_cnt * dim2,
		       iZ + thread_idx + block_idx * thread_cnt * dim2,
		       block_cnt, thread_cnt);
	  /*
	  ZLAHQR(rE + thread_idx + block_idx * thread_cnt * dim, iE + thread_idx + block_idx * thread_cnt * dim,
		 rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2,
		 dim, block_cnt, thread_cnt);
	  */

#ifndef CUDA
	}
    }
#endif	// ndef CUDA

}

global void test_diag(double *rM, double *iM, double real, double imag, int dim, int block_cnt, int thread_cnt)
{
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for
#endif  // OPENMP
  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA
	  int off = thread_idx + dim*dim * thread_cnt * block_idx;

	  set_diagonal(rM + off, iM + off, real, imag, dim, block_cnt, thread_cnt);

#ifndef CUDA
	}
    }
#endif  // ndef CUDA
}
	  

global void test_ztrevc(double *rV, double *iV, double *rZ, double *iZ, double *rwork, double *iwork,
			int dim, int dim2, int block_cnt, int thread_cnt)
{
#ifndef CUDA

#ifdef OPENMP
#pragma omp parallel for schedule(static, 8)
#endif	// OPENMP
  for(int block_idx=0; block_idx<block_cnt; ++block_idx)
    {
      for(int thread_idx=0; thread_idx<thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA
	  int off_matrix = thread_idx + block_idx * thread_cnt * dim2;
	  int off_vector = thread_idx + 2 * block_idx * thread_cnt * dim;
	  ztrevc(rV + thread_idx + block_idx * thread_cnt * dim2, iV + thread_idx + block_idx * thread_cnt * dim2,
		 0, 0,
		 rZ + thread_idx + block_idx * thread_cnt * dim2, iZ + thread_idx + block_idx * thread_cnt * dim2,
		 rwork + off_vector, iwork + off_vector,
		 dim, block_cnt, thread_cnt);
	  
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

global void test_backsubstitution(double *rT, double *iT, double *rx, double *ix,
				  int dim, int block_cnt, int thread_cnt)
{
#ifndef CUDA

#ifdef OPENMP
#pragma omp parallel for
#endif	// OPENMP
  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA
	  int dim2 = dim * dim;
	  int off_matrix = thread_idx + block_idx * thread_cnt * dim2;
	  int off_vector = thread_idx + block_idx * thread_cnt * dim;
	  backsubstitution(dim,
			   rT + off_matrix, iT + off_matrix, rx + off_vector, ix + off_vector,
			   dim, thread_cnt);
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

global void test_zgemv(bool hermitian, double *rA, double *iA, double *rx, double *ix, double rbeta, double ibeta,
		       double *ry, double *iy, int dim, int block_cnt, int thread_cnt)
{
  int dim2 = dim*dim;
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for
#endif	// OPENMP
  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA
	  zgemv_simple(hermitian, dim, dim,
		       rA + thread_idx + block_idx * thread_cnt * dim2,
		       iA + thread_idx + block_idx * thread_cnt * dim2,
		       dim,
		       rx + thread_idx + block_idx * thread_cnt * dim,
		       ix + thread_idx + block_idx * thread_cnt * dim,
		       rbeta, ibeta,
		       ry + thread_idx + block_idx * thread_cnt * dim,
		       iy + thread_idx + block_idx * thread_cnt * dim,
		       block_cnt, thread_cnt);
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

global void test_zgehrd(double *rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
			int dim, int block_cnt, int thread_cnt)
{
  int dim2 = dim * dim;
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for schedule(static, 8)
#endif  // OPENMP
  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif  // ndef CUDA
	  zgehrd(rA + thread_idx + block_idx * thread_cnt * dim2,
		 iA + thread_idx + block_idx * thread_cnt * dim2,
		 rtau + thread_idx + block_idx * thread_cnt * dim,
		 itau + thread_idx + block_idx * thread_cnt * dim,
		 rwork + thread_idx + block_idx * thread_cnt * dim,
		 iwork + thread_idx + block_idx * thread_cnt * dim,
		 dim, block_cnt, thread_cnt);
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

global void test_zunghr(double *rA, double *iA, double *rtau, double *itau, double *rwork, double *iwork,
			int dim, int block_cnt, int thread_cnt)
{
  int dim2 = dim*dim;
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for schedule(static, 8)
#endif	// OPENMP
  for(int block_idx = 0; block_idx < block_cnt; ++block_idx)
    {
      for(int thread_idx = 0; thread_idx < thread_cnt; ++thread_idx)
	{
#else
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// CUDA
	  zunghr(rA + thread_idx + block_idx * thread_cnt * dim2,
		 iA + thread_idx + block_idx * thread_cnt * dim2,
		 rtau + thread_idx + block_idx * thread_cnt * dim,
		 itau + thread_idx + block_idx * thread_cnt * dim,
		 rwork + thread_idx + block_idx * thread_cnt * dim,
		 iwork + thread_idx + block_idx * thread_cnt * dim,
		 dim, block_cnt, thread_cnt);
#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}

  
/*
global void test_ztrevc(double *rT, double *iT, double *rL, double *iL, double *rR, double *iR,
			int dim, int block_cnt, int thread_cnt)
{
  int dim2 = dim*dim;
#ifndef CUDA
#ifdef OPENMP
#pragma omp parallel for
#endif	// OPENMP
  for(int block_idx=0; block_idx<block_cnt; ++block_idx)
    {
      for(int thread_idx=0; thread_idx<thread_cnt; ++thread_idx)
	{
#else  // ndef CUDA
	  int block_idx = blockIdx.x;
	  int thread_idx = threadIdx.x;
#endif	// ndef CUDA

	  ztrevc(rT + thread_idx + block_idx * thread_cnt * dim2, iT + thread_idx + block_idx * thread_cnt * dim2,
		 rL + thread_idx + block_idx * thread_cnt * dim2, iL + thread_idx + block_idx * thread_cnt * dim2,
		 rR + thread_idx + block_idx * thread_cnt * dim2, iR + thread_idx + block_idx * thread_cnt * dim2,
		 dim, block_cnt, thread_cnt);

#ifndef CUDA
	}
    }
#endif	// ndef CUDA
}
*/
  
#endif // TOOLBOX_HPP_

