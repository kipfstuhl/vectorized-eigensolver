//============================================================================
// Name        : ComplexDouble.hpp
// Author      : J. Kipfstuhl, M. Presenhuber, M. Liebmann
// Version     : 2.0
// Copyright   : University of Graz
// Description : ZLAQHRV-Algorithm
//============================================================================

#ifndef COMPLEXDOUBLE_HPP_
#define COMPLEXDOUBLE_HPP_

#include <iostream>
#include <cmath>
#include <cfloat>

#include "definitions.hpp"

using namespace std;

cuda double dabs(double x)
{
    return fabs(x);
}

cuda double cdivex2(const double A, const double B, const double C, const double D, const double R, const double T)
{
    double BR;

    if (R != 0.0)
    {
        BR = B * R;
        if (BR != 0.0)
        {
            return (A + BR) * T;
        }
        else
        {
            return A * T + (B * T) * R;
        }
    }
    else
    {
        return (A + D * (B / C)) * T;
    }
}

cuda void cdivex1(const double A, const double B, const double C, const double D, double& P, double& Q)
{
    double R, T;

    R = D / C;
    T = 1.0 / (C + D * R);
    P = cdivex2(A, B, C, D, R, T);
    Q = cdivex2(B, -A, C, D, R, T);
}

cuda void cdivex(const double A, const double B, const double C, const double D, double& P, double& Q)
{
    double AA, BB, CC, DD, AB, CD, S, OV, UN, BE, EPS;
    AA = A;
    BB = B;
    CC = C;
    DD = D;
    AB = fmax(dabs(A), dabs(B));
    CD = fmax(dabs(C), dabs(D));
    S = 1.0;

    OV = DBL_MAX;
    UN = DBL_MIN;
    EPS = DBL_EPSILON * 0.5;
    BE = 2.0 / (EPS * EPS);

    if (AB >= 0.5 * OV)
    {
        AA = 0.5 * AA;
        BB = 0.5 * BB;
        S = 2.0 * S;
    }
    if (CD >= 0.5 * OV)
    {
        CC = 0.5 * CC;
        DD = 0.5 * DD;
        S = 0.5 * S;
    }
    if (AB <= UN * 2.0 / EPS)
    {
        AA = AA * BE;
        BB = BB * BE;
        S = S / BE;
    }
    if (CD <= UN * 2.0 / EPS)
    {
        CC = CC * BE;
        DD = DD * BE;
        S = S * BE;
    }
    if (dabs(D) <= dabs(C))
    {
        cdivex1(AA, BB, CC, DD, P, Q);
    }
    else
    {
        cdivex1(BB, AA, DD, CC, P, Q);
        Q = -Q;
    }
    P = P * S;
    Q = Q * S;
}

struct ComplexDouble
{

public:
    double r;
    double i;

    cuda ComplexDouble(double real, double imag)
    {
        r = real;
        i = imag;
    }
    cuda ComplexDouble()
    {

    }

    cuda
    ComplexDouble& operator+=(const ComplexDouble& rhs)
    {
        r += rhs.r;
        i += rhs.i;
        return *this;
    }

    cuda
    ComplexDouble& operator-=(const ComplexDouble& rhs)
    {
        r -= rhs.r;
        i -= rhs.i;
        return *this;
    }

    cuda
    ComplexDouble& operator*=(const ComplexDouble& rhs)
    {
        double lr = r, li = i;
        double rr = rhs.r, ri = rhs.i;
        r = lr * rr - li * ri;
        i = lr * ri + li * rr;
        return *this;
    }

    cuda
    ComplexDouble& operator/=(const ComplexDouble& rhs)
    {
        double ratio, den;
        double ar, ai, cr, ci, rr = rhs.r, ri = rhs.i;
        ar = dabs(rr);
        ai = dabs(ri);

        if (ar <= ai)
        {
            cr = rr;
            ci = ri;
        }
        else
        {
            ci = rr;
            cr = ri;
        }

        ratio = cr / ci;
        den = ci * (1.0 + ratio * ratio);

        if (ar <= ai)
        {
            cr = r * ratio + i;
            ci = i * ratio - r;
        }
        else
        {
            cr = r + i * ratio;
            ci = i - r * ratio;
        }

        r = cr / den;
        i = ci / den;

        return *this;
    }

    cuda
    ComplexDouble& cinv()
    {
        double ratio, den;
        double ar, ai, cr, ci, rr = r, ri = i;
        ar = dabs(rr);
        ai = dabs(ri);

        if (ar <= ai)
        {
            cr = rr;
            ci = ri;
        }
        else
        {
            ci = rr;
            cr = ri;
        }

        ratio = cr / ci;
        den = ci * (1.0 + ratio * ratio);
        cr = ratio / den;
        ci = 1.0 / den;

        if (ar <= ai)
        {
            r = cr;
            i = -ci;
        }
        else
        {
            r = ci;
            i = -cr;
        }

        return *this;
    }

    cuda
    ComplexDouble& operator*=(const double& rhs)
    {
        r *= rhs;
        i *= rhs;
        return *this;
    }

    cuda
    ComplexDouble& operator/=(const double& rhs)
    {
        r /= rhs;
        i /= rhs;
        return *this;
    }

    cuda
    ComplexDouble& csqrt()
    {
        double cr, ci, lr = r, li = i;
        double rho = sqrt((sqrt(lr * lr + li * li) + dabs(lr)) * 0.5);
        double tau = dabs(li) / (rho * 2.0);

        if (lr > 0.0)
        {
            cr = rho;
            ci = tau;
        }
        else
        {
            cr = tau;
            ci = rho;
        }

        if (li < 0.0)
        {
            ci = -ci;
        }

        r = cr;
        i = ci;

        return *this;
    }

    friend ostream& operator<<(ostream& os, ComplexDouble z)
    {
        os << z.r << (z.i >= 0.0 ? "+" : "") << z.i << "i";
        return os;
    }

};

cuda void cinv(double& r, double& i)
{
    double ratio, den;
    double ar, ai, cr, ci, rr = r, ri = i;
    ar = dabs(rr);
    ai = dabs(ri);

    if (ar <= ai)
    {
        cr = rr;
        ci = ri;
    }
    else
    {
        ci = rr;
        cr = ri;
    }

    ratio = cr / ci;
    den = ci * (1.0 + ratio * ratio);
    cr = ratio / den;
    ci = 1.0 / den;

    if (ar <= ai)
    {
        r = cr;
        i = -ci;
    }
    else
    {
        r = ci;
        i = -cr;
    }

}

cuda void cdiv(double& r, double& i, const double rr, const double ri)
{
    double ratio, den;
    double ar, ai, cr, ci;
    ar = dabs(rr);
    ai = dabs(ri);

    if (ar <= ai)
    {
        cr = rr;
        ci = ri;
    }
    else
    {
        ci = rr;
        cr = ri;
    }

    ratio = cr / ci;
    den = ci * (1.0 + ratio * ratio);

    if (ar <= ai)
    {
        cr = r * ratio + i;
        ci = i * ratio - r;
    }
    else
    {
        cr = r + i * ratio;
        ci = i - r * ratio;
    }

    r = cr / den;
    i = ci / den;
}

cuda void cadd(double& r, double& i, const double rr, const double ri)
{
    r += rr;
    i += ri;
}

cuda void csub(double& r, double& i, const double rr, const double ri)
{
    r -= rr;
    i -= ri;
}

cuda void cmul(double& r, double& i, const double rr, const double ri)
{
    double lr = r, li = i;
    r = lr * rr - li * ri;
    i = li * rr + lr * ri;
}

cuda void cmulc(double& r, double& i, const double rr, const double ri)
{
    double lr = r, li = i;
    r = lr * rr + li * ri;
    i = li * rr - lr * ri;
}

cuda void cmul(double& r, double& i, const double rr)
{
    r *= rr;
    i *= rr;
}

cuda void cdiv(double& r, double& i, const double rr)
{
    r /= rr;
    i /= rr;
}

cuda double dabs2(double x, double y)
{
    double cx, cy;
    x = dabs(x);
    y = dabs(y);
    if (x > y)
        cx = x, cy = y;
    else
        cx = y, cy = x;
    if (cx == 0.0)
        cy = 0.0;
    else
        cy /= cx;
    return cx * sqrt(1.0 + cy * cy);
}

cuda double dabs3(double x, double y, double z)
{
    double w;
    x = dabs(x);
    y = dabs(y);
    z = dabs(z);
    w = fmax(fmax(x, y), z);

    if (w == 0.0)
    {
        w = x + y + z;
    }
    else
    {
        x /= w;
        y /= w;
        z /= w;
        w *= sqrt(x * x + y * y + z * z);
    }

    return w;
}

cuda double cabs1(double r, double i)
{
    return dabs(r) + dabs(i);
}

extern "C" double mkl_serv_hypot(double r, double i);

cuda double cabs(double r, double i)
{

#ifdef MKL
    long double zr = r, zi = i;
    return sqrt(zr * zr + zi * zi);
#else
    return sqrt(r * r + i * i);
#endif

}

cuda void csqrt(double& r, double& i)
{
    double cr, ci;

#ifdef MKL
    long double zr, zi, lr = r, li = i;
    zr = sqrt(0.5 * (mkl_serv_hypot(lr, li) + dabs(lr)));
#else
    double zr, zi, lr = r, li = i;
    zr = sqrt(0.5 * (cabs(lr, li) + dabs(lr)));
#endif

    if (zr == 0.0)
    {
        zi = 0.0;
    }
    else
    {
        zi = dabs(li) / (2.0 * zr);
    }

    if (lr < 0.0)
    {
        cr = zi, ci = zr;
    }
    else
    {
        cr = zr, ci = zi;
    }

    if (li < 0.0)
    {
        ci = -ci;
    }

    r = cr;
    i = ci;
}

#endif
