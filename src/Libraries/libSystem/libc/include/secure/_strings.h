/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#ifndef _STRINGS_H_
# error "Never use <secure/_strings.h> directly; include <strings.h> instead."
#endif

#ifndef _SECURE__STRINGS_H_
#define _SECURE__STRINGS_H_

#include <sys/cdefs.h>
#include <Availability.h>
#include <secure/_common.h>

#if _USE_FORTIFY_LEVEL > 0

/* bcopy and bzero */

/* Removed in Issue 7 */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L

#if __has_builtin(__builtin___memmove_chk) || defined(__GNUC__)
#undef bcopy
/* void	bcopy(const void *src, void *dst, size_t len) */
#define bcopy(src, dest, ...) \
		__builtin___memmove_chk (dest, src, __VA_ARGS__, __darwin_obsz0 (dest))
#endif

#if __has_builtin(__builtin___memset_chk) || defined(__GNUC__)
#undef bzero
/* void	bzero(void *s, size_t n) */
#define bzero(dest, ...) \
		__builtin___memset_chk (dest, 0, __VA_ARGS__, __darwin_obsz0 (dest))
#endif

#endif

#endif /* _USE_FORTIFY_LEVEL > 0 */
#endif /* _SECURE__STRINGS_H_ */
