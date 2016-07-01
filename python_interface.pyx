

cimport numpy as np
import numpy as np
cimport cython
import cython
from libcpp cimport bool
ctypedef np.float64_t dtype_t

cimport openmp

cdef extern from "utils.cpp":
    void repMatrixSoA "repMatrixSoA"(int dim, int block_cnt, int thread_cnt, double* matrix, double* out) nogil
    void repVectorSoA "repVectorSoA"(int dim, int block_cnt, int thread_cnt, double *vec, double *out) nogil
    void rearrangeMatrices "rearrangeMatrices"(int dim, int block_cnt, int thread_cnt,
                                                   double *inputMat, double *outputMat) nogil
    void call_zlahqr "call_zlahqr_new"(int dim, int block_cnt, int thread_cnt, double *rE, double *iE,
                                           double *rM, double *iM, double *rS, double *iS) nogil
    void call_zgemv "call_zgemv"(bool hermitian, int dim, int block_cnt, int thread_cnt, double *rM, double *iM,
                                     double *rx, double *ix, double *ry, double *iy,
                                     double rbeta, double ibeta) nogil
    void call_ztrevc "call_ztrevc"(int dim, int block_cnt, int thread_cnt,
                                       double *rT, double *iT, double *rR, double *iR) nogil
    void call_backsubs "call_backsubstitution"(int dim, int block_cnt, int thread_cnt,
                                                   double *rT, double *iT, double *rb, double *ib) nogil
    void call_zgehrd "call_zgehrd"(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau) nogil
    void call_zunghr "call_zunghr"(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rtau, double *itau) nogil
    void hessenberg "hessenberg"(int dim, int block_cnt, int thread_cnt, double *rA, double *iA, double *rQ, double *iQ) nogil
    

def my_repMatrix(int dim, int block_cnt, int thread_cnt,
                     np.ndarray[dtype_t, ndim=1] matrix, np.ndarray[dtype_t, ndim=1] out):
    with nogil:
        repMatrixSoA(dim, block_cnt, thread_cnt, &matrix[0], &out[0])

def arrangeMatrices(int dim, int block_cnt, int thread_cnt,
                        np.ndarray[dtype_t, ndim=1] inputMat, np.ndarray[dtype_t, ndim=1] outputMat):
    with nogil:
        rearrangeMatrices(dim, block_cnt, thread_cnt, &inputMat[0], &outputMat[0])
        
def my_zlahqr(int dim, int block_cnt, int thread_cnt,
                  np.ndarray[dtype_t, ndim=1] rE, np.ndarray[dtype_t, ndim=1] iE,
                  np.ndarray[dtype_t, ndim=1] rM, np.ndarray[dtype_t, ndim=1] iM,
                  np.ndarray[dtype_t, ndim=1] rS, np.ndarray[dtype_t, ndim=1] iS):
    with nogil:
        call_zlahqr(dim, block_cnt, thread_cnt, &rE[0], &iE[0], &rM[0], &iM[0], &rS[0], &iS[0])

def my_backsubs(int dim, int block_cnt, int thread_cnt,
                    np.ndarray[dtype_t, ndim=1] rT, np.ndarray[dtype_t, ndim=1] iT,
                    np.ndarray[dtype_t, ndim=1] rb, np.ndarray[dtype_t, ndim=1] ib):
    with nogil:
        call_backsubs(dim, block_cnt, thread_cnt, &rT[0], &iT[0], &rb[0], &ib[0])

def my_repVector(int dim, int block_cnt, int thread_cnt,
                     np.ndarray[dtype_t, ndim=1] vector, np.ndarray[dtype_t, ndim=1] out):
    with nogil:
        repVectorSoA(dim, block_cnt, thread_cnt, &vector[0], &out[0])

def my_ztrevec(int dim, int block_cnt, int thread_cnt,
                   np.ndarray[dtype_t, ndim=1] rT, np.ndarray[dtype_t, ndim=1] iT,
                   np.ndarray[dtype_t, ndim=1] rR, np.ndarray[dtype_t, ndim=1] iR):
    with nogil:
        call_ztrevc(dim, block_cnt, thread_cnt, &rT[0], &iT[0], &rR[0], &iR[0])

def my_zgemv(bool hermitian, int dim, int block_cnt, int thread_cnt,
                 np.ndarray[dtype_t, ndim=1] rA, np.ndarray[dtype_t, ndim=1] iA,
                 np.ndarray[dtype_t, ndim=1] rx, np.ndarray[dtype_t, ndim=1] ix,
                 double rbeta, double ibeta,
                 np.ndarray[dtype_t, ndim=1] ry, np.ndarray[dtype_t, ndim=1] iy):
    with nogil:
        call_zgemv(hermitian, dim, block_cnt, thread_cnt,
                       &rA[0], &iA[0], &rx[0], &ix[0], &ry[0], &iy[0], rbeta, ibeta)

def my_zgehrd(int dim, int block_cnt, int thread_cnt,
                  np.ndarray[dtype_t, ndim=1] rA, np.ndarray[dtype_t, ndim=1] iA,
                  np.ndarray[dtype_t, ndim=1] rtau, np.ndarray[dtype_t, ndim=1] itau):
    with nogil:
        call_zgehrd(dim, block_cnt, thread_cnt, &rA[0], &iA[0], &rtau[0], &itau[0])

def my_zunghr(int dim, int block_cnt, int thread_cnt,
                  np.ndarray[dtype_t, ndim=1] rA, np.ndarray[dtype_t, ndim=1] iA,
                  np.ndarray[dtype_t, ndim=1] rtau, np.ndarray[dtype_t, ndim=1] itau):
    with nogil:
        call_zunghr(dim, block_cnt, thread_cnt, &rA[0], &iA[0], &rtau[0], &itau[0])

def my_hessenberg(int dim, int block_cnt, int thread_cnt,
                      np.ndarray[dtype_t, ndim=1] rA, np.ndarray[dtype_t, ndim=1] iA,
                      np.ndarray[dtype_t, ndim=1] rQ, np.ndarray[dtype_t, ndim=1] iQ):
    with nogil:
        hessenberg(dim, block_cnt, thread_cnt, &rA[0], &iA[0], &rQ[0], &iQ[0])
        
def extractMatrix(int dim, int thread_cnt, np.ndarray[dtype_t, ndim=1] matrix):
    return matrix[::thread_cnt][:dim*dim].reshape(dim,dim).T

def extractVector(int dim, int thread_cnt, np.ndarray[dtype_t, ndim=1] vector):
    return vector[::thread_cnt][:dim]

def test_backsubs(int dim, int block_cnt, int thread_cnt):
    a = np.random.random((dim, dim))
    a = np.triu(a)
    ra = np.zeros(dim*dim*thread_cnt*block_cnt, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim*dim), ra)
    
    x = np.random.random(dim)
    rx = np.zeros(dim*thread_cnt*block_cnt, dtype='double')
    ix = np.zeros_like(rx)
    my_repVector(dim, block_cnt, thread_cnt, x, rx)
    
    my_backsubs(dim, block_cnt, thread_cnt, ra, ia, rx, ix)

    sol = extractVector(dim, thread_cnt, rx)
    return a.dot(sol) - x

def test_zgehrd(int dim, int block_cnt, int thread_cnt):
    a = np.random.random((dim, dim))
    ra = np.zeros(dim*dim * thread_cnt * block_cnt, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim*dim), ra)

    rtau = np.zeros(dim * thread_cnt * block_cnt, dtype='double')
    itau = np.zeros_like(rtau)
    my_zgehrd(dim, block_cnt, thread_cnt, ra, ia, rtau, itau)
    erg = extractMatrix(dim, thread_cnt, ra) + 1.0j * extractMatrix(dim, thread_cnt, ia)
    return erg, a

def test_hessenberg(int dim, int block_cnt, int thread_cnt):
    a = np.random.random((dim, dim))
    ra = np.zeros(dim*dim * thread_cnt * block_cnt, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim*dim), ra)

    rq = np.zeros_like(ra)
    iq = np.zeros_like(rq)

    my_hessenberg(dim, block_cnt, thread_cnt, ra, ia, rq, iq)

    h = extractMatrix(dim, thread_cnt, ra) + 1.0j * extractMatrix(dim, thread_cnt, ia)
    q = extractMatrix(dim, thread_cnt, rq) + 1.0j * extractMatrix(dim, thread_cnt, iq)

    return h, q, a

def test_zgemv(int dim, int block_cnt, int thread_cnt, bool hermitian = False):
    a = np.random.random((dim,dim))
    ra = np.zeros(dim*dim*thread_cnt*block_cnt, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim*dim), ra)

    y = np.random.random(dim)
    ry = np.zeros(dim*thread_cnt*block_cnt, dtype='double')
    iy = np.zeros_like(ry)
    my_repVector(dim, block_cnt, thread_cnt, y, ry)

    x = np.random.random(dim)
    rx = np.zeros_like(iy)
    ix = np.zeros_like(rx)
    my_repVector(dim, block_cnt, thread_cnt, x, rx)

    my_zgemv(hermitian, dim, block_cnt, thread_cnt, ra, ia, rx, ix, 1.0, 0.0, ry, iy)
    axpy = extractVector(dim, thread_cnt, ry)

    if hermitian:
        sol = np.conj(a.T).dot(x) + y
    else:
        sol = a.dot(x) + y
    return np.abs(axpy - sol)

def test_zlahqr(int dim, int block_cnt, int thread_cnt):
    dim2 = dim * dim
    size = dim2 * block_cnt * thread_cnt
    sizev = dim * block_cnt * thread_cnt

    a = np.random.random((dim,dim))
    a = np.triu(a,-1)
    ra = np.zeros(size, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim2), ra)

    s = np.eye(dim)
    rschur = np.zeros(size, dtype='double')
    ischur = np.zeros(size, dtype='double')
    my_repMatrix(dim, block_cnt, thread_cnt, s.T.reshape(dim2), rschur)
    
    re = np.zeros(sizev, dtype='double')
    ie = np.zeros(sizev, dtype='double')

    my_zlahqr(dim, block_cnt, thread_cnt, re, ie, ra, ia, rschur, ischur)

    eigs = extractVector(dim, thread_cnt, re) + 1.0j * extractVector(dim, thread_cnt, ie)
    schurv = extractMatrix(dim, thread_cnt, rschur) + 1.0j * extractMatrix(dim, thread_cnt, ischur)
    t = extractMatrix(dim, thread_cnt, ra) + 1.0j * extractMatrix(dim, thread_cnt, ia)

    return (eigs, schurv, t, a)

def test_eigenvectors(int dim, int block_cnt, int thread_cnt):
    dim2 = dim * dim
    size = dim2 * block_cnt * thread_cnt
    sizev = dim * block_cnt * thread_cnt

    a = np.random.random((dim, dim))
    a = np.triu(a, -1)
    ra = np.zeros(size, dtype='double')
    ia = np.zeros_like(ra)
    my_repMatrix(dim, block_cnt, thread_cnt, a.T.reshape(dim2), ra)

    s = np.eye(dim)
    rsch = np.zeros(size, dtype='double')
    isch = np.zeros(size, dtype='double')
    my_repMatrix(dim, block_cnt, thread_cnt, s.T.reshape(dim2), rsch)

    re = np.zeros(sizev, dtype='double')
    ie = np.zeros(sizev, dtype='double')

    my_zlahqr(dim, block_cnt, thread_cnt, re, ie, ra, ia, rsch, isch)
    my_ztrevec(dim, block_cnt, thread_cnt, ra, ia, rsch, isch)
    
    eigs = extractVector(dim, thread_cnt, re) + 1.0j * extractVector(dim, thread_cnt, ie)
    vecs = extractMatrix(dim, thread_cnt, rsch) + 1.0j * extractMatrix(dim, thread_cnt, isch)

    return(eigs, vecs, a)
