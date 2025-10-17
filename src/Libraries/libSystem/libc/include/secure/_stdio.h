/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#ifndef _STDIO_H_
 #error error "Never use <secure/_stdio.h> directly; include <stdio.h> instead."
#endif

#ifndef _SECURE__STDIO_H_
#define _SECURE__STDIO_H_

#include <secure/_common.h>

#if _USE_FORTIFY_LEVEL > 0

#ifndef __has_builtin
#define _undef__has_builtin
#define __has_builtin(x) 0
#endif

#ifndef UNIFDEF_DRIVERKIT
/* sprintf, vsprintf, snprintf, vsnprintf */
#if __has_builtin(__builtin___sprintf_chk) || defined(__GNUC__)
extern int __sprintf_chk (char * __restrict, int, size_t,
			  const char * __restrict, ...);

#undef sprintf
#define sprintf(str, ...) \
  __builtin___sprintf_chk (str, 0, __darwin_obsz(str), __VA_ARGS__)
#endif
#endif /* UNIFDEF_DRIVERKIT */

#if __DARWIN_C_LEVEL >= 200112L
#if __has_builtin(__builtin___snprintf_chk) || defined(__GNUC__)
extern int __snprintf_chk (char * __restrict, size_t, int, size_t,
			   const char * __restrict, ...);

#undef snprintf
#define snprintf(str, len, ...) \
  __builtin___snprintf_chk (str, len, 0, __darwin_obsz(str), __VA_ARGS__)
#endif

#ifndef UNIFDEF_DRIVERKIT
#if __has_builtin(__builtin___vsprintf_chk) || defined(__GNUC__)
extern int __vsprintf_chk (char * __restrict, int, size_t,
			   const char * __restrict, va_list);

#undef vsprintf
#define vsprintf(str, format, ap) \
  __builtin___vsprintf_chk (str, 0, __darwin_obsz(str), format, ap)
#endif
#endif /* UNIFDEF_DRIVERKIT */

#if __has_builtin(__builtin___vsnprintf_chk) || defined(__GNUC__)
extern int __vsnprintf_chk (char * __restrict, size_t, int, size_t,
			    const char * __restrict, va_list);

#undef vsnprintf
#define vsnprintf(str, len, format, ap) \
  __builtin___vsnprintf_chk (str, len, 0, __darwin_obsz(str), format, ap)
#endif

#endif /* __DARWIN_C_LEVEL >= 200112L */

#ifdef _undef__has_builtin
#undef _undef__has_builtin
#undef __has_builtin
#endif

#endif /* _USE_FORTIFY_LEVEL > 0 */
#endif /* _SECURE__STDIO_H_ */
