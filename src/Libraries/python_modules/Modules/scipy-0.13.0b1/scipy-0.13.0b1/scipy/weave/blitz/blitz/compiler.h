/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#ifndef BZ_COMPILER_H
#define BZ_COMPILER_H

// The file <blitz/bzconfig.h> is used to select a compiler-specific
// config.h file that is generated automatically by configure.

#include <blitz/bzconfig.h>

/*
 * Define some kludges.
 */

#ifndef BZ_HAVE_TEMPLATES
    #error  In <blitz/config.h>: A working template implementation is required by Blitz++ (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_MEMBER_TEMPLATES
  #error  In <blitz/config.h>: Your compiler does not support member templates.  (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_FULL_SPECIALIZATION_SYNTAX
  #error In <blitz/config.h>: Your compiler does not support template<> full specialization syntax.  You may need to rerun the compiler/bzconfig script.
#endif

#ifndef BZ_HAVE_PARTIAL_ORDERING
  #error In <blitz/config.h>: Your compiler does not support partial ordering (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_PARTIAL_SPECIALIZATION
  #error In <blitz/config.h>: Your compiler does not support partial specialization (you may need to rerun the compiler/bzconfig script)
#endif

#ifdef BZ_HAVE_NAMESPACES
    #define BZ_NAMESPACE(X)        namespace X {
    #define BZ_NAMESPACE_END       }
    #define BZ_USING_NAMESPACE(X)  using namespace X;
#else
    #define BZ_NAMESPACE(X)
    #define BZ_NAMESPACE_END
    #define BZ_USING_NAMESPACE(X)
#endif

#ifdef BZ_HAVE_TEMPLATE_QUALIFIED_RETURN_TYPE
  #define BZ_USE_NUMTRAIT
#endif

#ifdef BZ_HAVE_DEFAULT_TEMPLATE_PARAMETERS
    #define BZ_TEMPLATE_DEFAULT(X)   = X
#else
    #define BZ_TEMPLATE_DEFAULT
#endif

#ifndef BZ_HAVE_EXPLICIT
    #define explicit   
#endif

#ifdef BZ_HAVE_TYPENAME
    #define _bz_typename     typename
#else
    #define _bz_typename
#endif

#ifndef BZ_HAVE_MUTABLE
    #define mutable
#endif

#ifdef BZ_DISABLE_RESTRICT
 #undef BZ_HAVE_NCEG_RESTRICT
#endif

#ifndef BZ_HAVE_NCEG_RESTRICT
    #if defined(BZ_HAVE_NCEG_RESTRICT_EGCS)
        #define restrict     __restrict__
    #else
        #define restrict
    #endif
#endif

#if !defined(BZ_HAVE_BOOL) && !defined(BZ_NO_BOOL_KLUDGE)
    #define bool    int
    #define true    1
    #define false   0
#endif

#ifdef BZ_HAVE_ENUM_COMPUTATIONS_WITH_CAST
    #define BZ_ENUM_CAST(X)   (int)X
#elif defined(BZ_HAVE_ENUM_COMPUTATIONS)
    #define BZ_ENUM_CAST(X)   X
#else
    #error In <blitz/config.h>: Your compiler does not support enum computations.  You may have to rerun compiler/bzconfig.
#endif

#if defined(BZ_MATH_FN_IN_NAMESPACE_STD)
  #define BZ_MATHFN_SCOPE(x) std::x
#elif defined(BZ_HAVE_NAMESPACES)
  #define BZ_MATHFN_SCOPE(x) ::x
#else
  #define BZ_MATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_COMPLEX_MATH_IN_NAMESPACE_STD)
  #define BZ_CMATHFN_SCOPE(x) std::x
#elif defined(BZ_HAVE_NAMESPACES)
  #define BZ_CMATHFN_SCOPE(x) ::x
#else
  #define BZ_CMATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_NAMESPACES)
  #define BZ_IEEEMATHFN_SCOPE(x) ::x
#else
  #define BZ_IEEEMATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_NAMESPACES)
  #define BZ_BLITZ_SCOPE(x) blitz::x
#else
  #define BZ_BLITZ_SCOPE(x) ::x
#endif

#if defined(BZ_HAVE_NAMESPACES) && defined(BZ_HAVE_STD)
  #define BZ_STD_SCOPE(x) std::x
#else
  #define BZ_STD_SCOPE(x) ::x
#endif

#endif // BZ_COMPILER_H

