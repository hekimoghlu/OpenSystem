/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include <complex>

#include "_faddeeva.h"

using namespace std;

EXTERN_C_START

npy_cdouble faddeeva_w(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    std::complex<double> w = Faddeeva::w(z);
    return npy_cpack(real(w), imag(w));
}

npy_cdouble faddeeva_erf(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erf(z);
    return npy_cpack(real(w), imag(w));
}

npy_cdouble faddeeva_erfc(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfc(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfcx(double x)
{
    return Faddeeva::erfcx(x);
}

npy_cdouble faddeeva_erfcx_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfcx(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfi(double x)
{
    return Faddeeva::erfi(x);
}

npy_cdouble faddeeva_erfi_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfi(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_dawsn(double x)
{
    return Faddeeva::Dawson(x);
}

npy_cdouble faddeeva_dawsn_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::Dawson(z);
    return npy_cpack(real(w), imag(w));
}

EXTERN_C_END
