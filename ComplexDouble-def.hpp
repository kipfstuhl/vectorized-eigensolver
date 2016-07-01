
#ifndef COMPLEXDOUBLE_DEF_HPP_
#define COMPLEXDOUBLE_DEF_HPP_

#include <iostream>
#include <cmath>
#include <cfloat>


struct ComplexDouble
{/*
public:
  double r;
  double i;
  
  cuda ComplexDouble(double real, double imag);

  cuda ComplexDouble();

  cuda
  ComplexDouble& operator+=(const ComplexDouble& rhs);
  
  cuda
  ComplexDouble& operator-=(const ComplexDouble& rhs);

  cuda
  ComplexDouble& operator*=(const ComplexDouble& rhs);
  
  cuda
  ComplexDouble& operator/=(const ComplexDouble& rhs);
  
  cuda
  ComplexDouble& cinv();
  
  cuda
  ComplexDouble& operator*=(const double& rhs);
  
  cuda
  ComplexDouble& operator/=(const double& rhs);
  
  cuda
  ComplexDouble& csqrt();
  
  friend ostream& operator<<(ostream& os, ComplexDouble z);
 */
};

cuda double dabs(double x);

cuda double cdivex2(const double A, const double B, const double C, const double D, const double R, const double T);

cuda void cdivex1(const double A, const double B, const double C, const double D, double& P, double& Q);

cuda void cdivex(const double A, const double B, const double C, const double D, double& P, double& Q);

cuda void cinv(double& r, double& i);

cuda void cdiv(double& r, double& i, const double rr, const double ri);

cuda void cadd(double& r, double& i, const double rr, const double ri);

cuda void csub(double& r, double& i, const double rr, const double ri);

cuda void cmul(double& r, double& i, const double rr, const double ri);

cuda void cmulc(double& r, double& i, const double rr, const double ri);

cuda void cmul(double& r, double& i, const double rr);

cuda void cdiv(double& r, double& i, const double rr);

cuda double dabs2(double x, double y);

cuda double dabs3(double x, double y, double z);

cuda double cabs1(double r, double i);

extern "C" double mkl_serv_hypot(double r, double i);

cuda double cabs(double r, double i);

cuda void csqrt(double& r, double& i);

#endif	// COMPLEXDOUBLE_DEF_HPP_
