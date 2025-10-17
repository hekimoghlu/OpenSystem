/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#ifndef _XLOCALE__STDLIB_H_
#define _XLOCALE__STDLIB_H_

#include <_bounds.h>
#include <sys/cdefs.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_wchar_t.h>
#include <__xlocale.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS
double	 atof_l(const char *, locale_t);
int	 atoi_l(const char *, locale_t);
long	 atol_l(const char *, locale_t);
#if !__DARWIN_NO_LONG_LONG
long long
	 atoll_l(const char *, locale_t);
#endif /* !__DARWIN_NO_LONG_LONG */
int	 mblen_l(const char *_LIBC_COUNT(__n), size_t __n, locale_t);
size_t	 mbstowcs_l(wchar_t * __restrict _LIBC_COUNT(__n),
	    const char * __restrict, size_t __n, locale_t);
int	 mbtowc_l(wchar_t * __restrict _LIBC_UNSAFE_INDEXABLE,
	    const char * __restrict _LIBC_COUNT(__n), size_t __n, locale_t);
double	 strtod_l(const char *, char *_LIBC_CSTR *, locale_t) __DARWIN_ALIAS(strtod_l);
float	 strtof_l(const char *, char *_LIBC_CSTR *, locale_t) __DARWIN_ALIAS(strtof_l);
long	 strtol_l(const char *, char *_LIBC_CSTR *, int, locale_t);
long double
	 strtold_l(const char *, char *_LIBC_CSTR *, locale_t);
long long
	 strtoll_l(const char *, char *_LIBC_CSTR *, int, locale_t);
#if !__DARWIN_NO_LONG_LONG
long long
	 strtoq_l(const char *, char *_LIBC_CSTR *, int, locale_t);
#endif /* !__DARWIN_NO_LONG_LONG */
unsigned long
	 strtoul_l(const char *, char *_LIBC_CSTR *, int, locale_t);
unsigned long long
	 strtoull_l(const char *, char *_LIBC_CSTR *, int, locale_t);
#if !__DARWIN_NO_LONG_LONG
unsigned long long
	 strtouq_l(const char *, char *_LIBC_CSTR *, int, locale_t);
#endif /* !__DARWIN_NO_LONG_LONG */
size_t	 wcstombs_l(char * __restric _LIBC_COUNT(__n),
	    const wchar_t * __restrict, size_t __n, locale_t);
int	 wctomb_l(char *, wchar_t, locale_t);

/* Poison the following routines if -fshort-wchar is set */
#if !defined(__cplusplus) && defined(__WCHAR_MAX__) && __WCHAR_MAX__ <= 0xffffU
#pragma GCC poison mbstowcs_l mbtowc_l wcstombs_l wctomb_l
#endif
__END_DECLS

#endif /* _XLOCALE__STDLIB_H_ */
