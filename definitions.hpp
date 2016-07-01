
#ifndef DEFINITIONS_HPP_
#define DEFINITIONS_HPP_

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

//#include "ComplexDouble-def.hpp"
//#incldue "Toolbox-def.hpp"


#endif	// DEFINITIONS_HPP_
