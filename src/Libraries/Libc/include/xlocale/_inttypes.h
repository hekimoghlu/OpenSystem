/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#ifndef _XLOCALE__INTTYPES_H_
#define _XLOCALE__INTTYPES_H_

#include <sys/cdefs.h>
#include <_bounds.h>
#include <stdint.h>
#include <stddef.h> /* wchar_t */
#include <__xlocale.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS
intmax_t  strtoimax_l(const char * __restrict nptr,
		char *_LIBC_CSTR * __restrict endptr, int base, locale_t);
uintmax_t strtoumax_l(const char * __restrict nptr,
		char *_LIBC_CSTR * __restrict endptr, int base, locale_t);
intmax_t  wcstoimax_l(const wchar_t * __restrict nptr,
		wchar_t *_LIBC_CSTR * __restrict endptr, int base, locale_t);
uintmax_t wcstoumax_l(const wchar_t * __restrict nptr,
		wchar_t *_LIBC_CSTR * __restrict endptr, int base, locale_t);

/* Poison the following routines if -fshort-wchar is set */
#if !defined(__cplusplus) && defined(__WCHAR_MAX__) && __WCHAR_MAX__ <= 0xffffU
#pragma GCC poison wcstoimax_l wcstoumax_l
#endif
__END_DECLS

#endif /* _XLOCALE__INTTYPES_H_ */
