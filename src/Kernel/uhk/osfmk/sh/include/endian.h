/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
/*	$NetBSD: endian.h,v 1.4 2000/03/17 00:09:25 mycroft Exp $	*/

/* Written by Manuel Bouyer. Public domain */

#ifndef _SH_ENDIAN_H_
#define	_SH_ENDIAN_H_

#ifndef __FROM_SYS__ENDIAN
#include <sys/_types.h>
#endif

static __inline __uint16_t
__swap16md(__uint16_t _x)
{
	__uint16_t _rv;

	__asm volatile ("swap.b %1,%0" : "=r"(_rv) : "r"(_x));

	 return (_rv);
}

static __inline __uint32_t
__swap32md(__uint32_t _x)
{
	__uint32_t _rv;

	__asm volatile ("swap.b %1,%0; swap.w %0,%0; swap.b %0,%0"
			  : "=r"(_rv) : "r"(_x));

	return (_rv);
}

static __inline __uint64_t
__swap64md(__uint64_t _x)
{
	__uint64_t _rv;

	_rv = (__uint64_t)__swap32md((__uint32_t)(_x >> 32)) |
	    (__uint64_t)__swap32md((__uint32_t)_x) << 32;

	return (_rv);
}

/* Tell sys/endian.h we have MD variants of the swap macros.  */
#define __HAVE_MD_SWAP

#ifdef __LITTLE_ENDIAN__
#define	_BYTE_ORDER _LITTLE_ENDIAN
#else
#define	_BYTE_ORDER _BIG_ENDIAN
#endif
#define	__STRICT_ALIGNMENT

#ifndef __FROM_SYS__ENDIAN
#include <sys/endian.h>
#endif

#endif /* !_SH_ENDIAN_H_ */
