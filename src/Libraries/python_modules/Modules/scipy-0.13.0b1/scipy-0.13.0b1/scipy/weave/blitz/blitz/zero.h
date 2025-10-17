/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
/*
 * The purpose of the ZeroElement class is to provide an lvalue for
 * non-const element access of matrices with zero elements.  For
 * example, a tridiagonal matrix has many elements which are
 * always zero:
 *
 * [ x x 0 0 ]
 * [ x x x 0 ]
 * [ 0 x x x ]
 * [ 0 0 x x ]
 *
 * To implement an operator()(int i, int j) for a tridiagonal
 * matrix which may be used as an lvalue
 *
 * e.g. Matrix<double, Tridiagonal> M(4,4);
 *      M(1,2) = 3.0L;
 *
 * some way of returning an lvalue for the zero elements is needed.
 * (Either that, or an intermediate class must be returned -- but
 * this is less efficient).  The solution used for the Blitz++
 * library is to have a unique zero element for each numeric
 * type (float, double, etc.).  This zero element is then
 * returned as an lvalue when needed.
 *
 * The disadvantage is the possibility of setting the global
 * zero-element to something non-zero.  
 */

#ifndef BZ_ZERO_H
#define BZ_ZERO_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_numtype>
class ZeroElement {
public:
    typedef P_numtype T_numtype;

    static T_numtype& zero()
    { 
        return zero_; 
    }

private:
    static T_numtype zero_;
};

// Specialization of ZeroElement for complex<float>, complex<double>,
// and complex<long double>

#define BZZERO_DECLARE(T)            \
  template<>                         \
  class ZeroElement<T > {            \
  public:                            \
    static T& zero()                 \
    { return zero_; }                \
  private:                           \
    static T zero_;                  \
  }

#ifdef BZ_HAVE_COMPLEX
  BZZERO_DECLARE(complex<float>);
  BZZERO_DECLARE(complex<double>);
  BZZERO_DECLARE(complex<long double>);
#endif // BZ_HAVE_COMPLEX

// initialization of static data member for general class template

template<typename P_numtype>
P_numtype ZeroElement<P_numtype>::zero_ = 0;

BZ_NAMESPACE_END

#endif // BZ_ZERO_H

