/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef _I386_LIMITS_H_
#define _I386_LIMITS_H_

#if defined (__i386__) || defined (__x86_64__)

#include <sys/cdefs.h>
#ifndef DRIVERKIT
#include <i386/_limits.h>
#endif

#if defined(KERNEL)
#ifdef XNU_KERNEL_PRIVATE
/*
 * Xcode doesn't currently set up search paths correctly for Kernel extensions,
 * so the clang headers are not seen in the correct order to use their limits.
 */
#endif
#define USE_CLANG_LIMITS 0
#else
#if defined(__has_feature) && __has_feature(modules)
#define USE_CLANG_LIMITS 1
#else
#define USE_CLANG_LIMITS 0
#endif
#endif

#undef  MB_LEN_MAX
#define MB_LEN_MAX      6               /* Allow 31 bit UTF2 */

#ifndef DRIVERKIT
#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
#define CLK_TCK         __DARWIN_CLK_TCK        /* ticks per second */
#endif /* !_ANSI_SOURCE && (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */
#endif

#if (defined(__has_include) && __has_include(<__xnu_libcxx_sentinel.h>))

#if !__has_include_next(<limits.h>)
#error Do not build with -nostdinc (use GCC_USE_STANDARD_INCLUDE_SEARCHING=NO)
#else
#include_next <limits.h>
#endif /* __has_include_next */

#elif !USE_CLANG_LIMITS

#define CHAR_BIT        8               /* number of bits in a char */

/*
 * According to ANSI (section 2.2.4.2), the values below must be usable by
 * #if preprocessing directives.  Additionally, the expression must have the
 * same type as would an expression that is an object of the corresponding
 * type converted according to the integral promotions.  The subtraction for
 * INT_MIN and LONG_MIN is so the value is not unsigned; 2147483648 is an
 * unsigned int for 32-bit two's complement ANSI compilers (section 3.1.3.2).
 * These numbers work for pcc as well.  The UINT_MAX and ULONG_MAX values
 * are written as hex so that GCC will be quiet about large integer constants.
 */
#define SCHAR_MAX       127             /* min value for a signed char */
#define SCHAR_MIN       (-128)          /* max value for a signed char */

#define UCHAR_MAX       255             /* max value for an unsigned char */
#define CHAR_MAX        127             /* max value for a char */
#define CHAR_MIN        (-128)          /* min value for a char */

#define USHRT_MAX       65535           /* max value for an unsigned short */
#define SHRT_MAX        32767           /* max value for a short */
#define SHRT_MIN        (-32768)        /* min value for a short */

#define UINT_MAX        0xffffffff      /* max value for an unsigned int */
#define INT_MAX         2147483647      /* max value for an int */
#define INT_MIN         (-2147483647-1) /* min value for an int */

#ifdef __LP64__
#define ULONG_MAX       0xffffffffffffffffUL    /* max unsigned long */
#define LONG_MAX        0x7fffffffffffffffL     /* max signed long */
#define LONG_MIN        (-0x7fffffffffffffffL-1) /* min signed long */
#else /* !__LP64__ */
#define ULONG_MAX       0xffffffffUL    /* max unsigned long */
#define LONG_MAX        2147483647L     /* max signed long */
#define LONG_MIN        (-2147483647L-1) /* min signed long */
#endif /* __LP64__ */

#define ULLONG_MAX      0xffffffffffffffffULL   /* max unsigned long long */
#define LLONG_MAX       0x7fffffffffffffffLL    /* max signed long long */
#define LLONG_MIN       (-0x7fffffffffffffffLL-1) /* min signed long long */

#endif /* !USE_CLANG_LIMITS */

#if !defined(_ANSI_SOURCE)
#ifdef __LP64__
#define LONG_BIT        64
#else /* !__LP64__ */
#define LONG_BIT        32
#endif /* __LP64__ */
#define SSIZE_MAX       LONG_MAX        /* max value for a ssize_t */
#define WORD_BIT        32

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE)
#define SIZE_T_MAX      ULONG_MAX       /* max value for a size_t */

#define UQUAD_MAX       ULLONG_MAX
#define QUAD_MAX        LLONG_MAX
#define QUAD_MIN        LLONG_MIN

#endif /* (!_POSIX_C_SOURCE && !_XOPEN_SOURCE) || _DARWIN_C_SOURCE */
#endif /* !_ANSI_SOURCE */

#undef USE_CLANG_LIMITS

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif /* _I386_LIMITS_H_ */
