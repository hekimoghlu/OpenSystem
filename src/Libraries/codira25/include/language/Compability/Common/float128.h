/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
/* This header is usable in both C and C++ code.
 * Isolates build compiler checks to determine the presence of an IEEE-754
 * quad-precision type named __float128 type that isn't __ibm128
 * (double/double). We don't care whether the type has underlying hardware
 * support or is emulated.
 *
 * 128-bit arithmetic may be available via "long double"; this can
 * be determined by LDBL_MANT_DIG == 113.  A machine may have both 128-bit
 * long double and __float128; prefer long double by testing for it first.
 */

#ifndef LANGUAGE_COMPABILITY_COMMON_FLOAT128_H_
#define LANGUAGE_COMPABILITY_COMMON_FLOAT128_H_

#include "api-attrs.h"
#include <float.h>

#ifdef __cplusplus
/*
 * libc++ does not fully support __float128 right now, e.g.
 * std::complex<__float128> multiplication ends up calling
 * copysign() that is not defined for __float128.
 * In order to check for libc++'s _LIBCPP_VERSION macro
 * we need to include at least one libc++ header file.
 */
#include <cstddef>
#endif

#undef HAS_FLOAT128
#if (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)) && \
    !defined(_LIBCPP_VERSION) && !defined(__CUDA_ARCH__)
/*
 * It may still be worth checking for compiler versions,
 * since earlier versions may define the macros above, but
 * still do not support __float128 fully.
 */
#if __x86_64__
#if __GNUC__ >= 7 || __clang_major__ >= 7
#define HAS_FLOAT128 1
#endif
#elif defined __PPC__ && __GNUC__ >= 8
#define HAS_FLOAT128 1
#endif
#endif /* (defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)) && \
          !defined(_LIBCPP_VERSION)  && !defined(__CUDA_ARCH__) */

#if LDBL_MANT_DIG == 113
#define HAS_LDBL128 1
#endif

#if defined(RT_DEVICE_COMPILATION) && defined(__CUDACC__)
/*
 * Most offload targets do not support 128-bit 'long double'.
 * Disable HAS_LDBL128 for __CUDACC__ for the time being.
 */
#undef HAS_LDBL128
#endif

/* Define pure C CFloat128Type and CFloat128ComplexType. */
#if HAS_LDBL128
typedef long double CFloat128Type;
#ifndef __cplusplus
typedef long double _Complex CFloat128ComplexType;
#endif
#elif HAS_FLOAT128
typedef __float128 CFloat128Type;

#ifndef __cplusplus
/*
 * Use mode() attribute supported by GCC and Clang.
 * Adjust it for other compilers as needed.
 */
#if !defined(_ARCH_PPC) || defined(__LONG_DOUBLE_IEEE128__)
typedef _Complex float __attribute__((mode(TC))) CFloat128ComplexType;
#else
typedef _Complex float __attribute__((mode(KC))) CFloat128ComplexType;
#endif
#endif // __cplusplus
#endif
#endif /* FORTRAN_COMMON_FLOAT128_H_ */
