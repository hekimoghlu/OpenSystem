/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

// -*- C++ -*-
/***************************************************************************
 * blitz/numtrait.h      Declaration of the NumericTypeTraits class
 *
 * $Id: numtrait.h 1414 2005-11-01 22:04:59Z cookedm $
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This code was relicensed under the modified BSD license for use in SciPy
 * by Todd Veldhuizen (see LICENSE.txt in the weave directory).
 *
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/

#ifndef BZ_NUMTRAIT_H
#define BZ_NUMTRAIT_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

#ifndef BZ_USE_NUMTRAIT
  #define BZ_SUMTYPE(X)    X
  #define BZ_DIFFTYPE(X)   X
  #define BZ_FLOATTYPE(X)  X
  #define BZ_SIGNEDTYPE(X) X
#else

#define BZ_SUMTYPE(X)   _bz_typename NumericTypeTraits<X>::T_sumtype
#define BZ_DIFFTYPE(X)  _bz_typename NumericTypeTraits<X>::T_difftype
#define BZ_FLOATTYPE(X) _bz_typename NumericTypeTraits<X>::T_floattype
#define BZ_SIGNEDTYPE(X) _bz_typename NumericTypeTraits<X>::T_signedtype

template<typename P_numtype>
class NumericTypeTraits {
public:
    typedef P_numtype T_sumtype;    // Type to be used for summing
    typedef P_numtype T_difftype;   // Type to be used for difference
    typedef P_numtype T_floattype;  // Type to be used for floating-point
                                    // calculations
    typedef P_numtype T_signedtype; // Type to be used for signed calculations
    enum { hasTrivialCtor = 0 };    // Assume the worst
};

#define BZDECLNUMTRAIT(X,Y,Z,W,U)                                   \
    template<>                                                      \
    class NumericTypeTraits<X> {                                    \
    public:                                                         \
        typedef Y T_sumtype;                                        \
        typedef Z T_difftype;                                       \
        typedef W T_floattype;                                      \
        typedef U T_signedtype;                                     \
        enum { hasTrivialCtor = 1 };                                \
    }                                                               

#ifdef BZ_HAVE_BOOL
    BZDECLNUMTRAIT(bool,unsigned,int,float,int);
#endif

BZDECLNUMTRAIT(char,int,int,float,char);
BZDECLNUMTRAIT(unsigned char, unsigned, int, float,int);
BZDECLNUMTRAIT(short int, int, int, float, short int);
BZDECLNUMTRAIT(short unsigned int, unsigned int, int, float, int);
BZDECLNUMTRAIT(int, long, int, float, int);
BZDECLNUMTRAIT(unsigned int, unsigned long, int, float, long);
BZDECLNUMTRAIT(long, long, long, double, long);
BZDECLNUMTRAIT(unsigned long, unsigned long, long, double, long);
BZDECLNUMTRAIT(float, double, float, float, float);
BZDECLNUMTRAIT(double, double, double, double, double);

#ifdef BZ_HAVE_COMPLEX
// BZDECLNUMTRAIT(complex<float>, complex<double>, complex<float>, complex<float>);
// BZDECLNUMTRAIT(complex<double>, complex<long double>, complex<double>, complex<double>);
#endif // BZ_HAVE_COMPLEX

#endif // BZ_USE_NUMTRAIT

BZ_NAMESPACE_END

#endif // BZ_NUMTRAIT_H
