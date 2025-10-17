/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#if (defined(__has_include) && __has_include(<__xnu_libcxx_sentinel.h>) && !defined(XNU_LIBCXX_SDKROOT))

#if !__has_include_next(<stddef.h>)
#error Do not build with -nostdinc (use GCC_USE_STANDARD_INCLUDE_SEARCHING=NO)
#endif /* __has_include_next */

#include_next <stddef.h>

#else /* (defined(__has_include) && __has_include(<__xnu_libcxx_sentinel.h>) && !defined(XNU_LIBCXX_SDKROOT)) */

#ifndef __STDDEF_H
#define __STDDEF_H

#undef NULL
#ifdef __cplusplus
#if __cplusplus >= 201103L
#define NULL nullptr
#else
#undef __null  // VC++ hack.
#define NULL __null
#endif
#else
#define NULL ((void*)0)
#endif

#ifndef _PTRDIFF_T
#define _PTRDIFF_T
typedef __typeof__(((int*)NULL) - ((int*)NULL)) ptrdiff_t;
#endif
#ifndef _SIZE_T
#define _SIZE_T
typedef __typeof__(sizeof(int)) size_t;
#endif
#ifndef __cplusplus
#ifndef _WCHAR_T
#define _WCHAR_T
typedef __WCHAR_TYPE__ wchar_t;
#endif
#endif

#ifndef offsetof
#define offsetof(t, d) __builtin_offsetof(t, d)
#endif

#endif /* __STDDEF_H */

/* Some C libraries expect to see a wint_t here. Others (notably MinGW) will
 * use __WINT_TYPE__ directly; accommodate both by requiring __need_wint_t
 */
#if defined(__need_wint_t)
#if !defined(_WINT_T)
#define _WINT_T
typedef __WINT_TYPE__ wint_t;
#endif /* _WINT_T */
#undef __need_wint_t
#endif /* __need_wint_t */

#endif
