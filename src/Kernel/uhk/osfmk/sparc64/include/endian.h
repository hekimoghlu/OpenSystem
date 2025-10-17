/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#ifndef _MACHINE_ENDIAN_H_
#define _MACHINE_ENDIAN_H_

#define	_BYTE_ORDER _BIG_ENDIAN

#ifdef _KERNEL

#define __ASI_P_L	0x88 /* == ASI_PRIMARY_LITTLE */

static inline __uint16_t
__mswap16(volatile const __uint16_t *m)
{
	__uint16_t v;

	__asm("lduha [%1] %2, %0 ! %3"
	    : "=r" (v)
	    : "r" (m), "n" (__ASI_P_L), "m" (*m));

	return (v);
}

static inline __uint32_t
__mswap32(volatile const __uint32_t *m)
{
	__uint32_t v;

	__asm("lduwa [%1] %2, %0 ! %3"
	    : "=r" (v)
	    : "r" (m), "n" (__ASI_P_L), "m" (*m));

	return (v);
}

static inline __uint64_t
__mswap64(volatile const __uint64_t *m)
{
	__uint64_t v;

	__asm("ldxa [%1] %2, %0 ! %3"
	    : "=r" (v)
	    : "r" (m), "n" (__ASI_P_L), "m" (*m));

	return (v);
}

static inline void
__swapm16(volatile __uint16_t *m, __uint16_t v)
{
	__asm("stha %1, [%2] %3 ! %0"
	    : "=m" (*m)
	    : "r" (v), "r" (m), "n" (__ASI_P_L));
}

static inline void
__swapm32(volatile __uint32_t *m, __uint32_t v)
{
	__asm("stwa %1, [%2] %3 ! %0"
	    : "=m" (*m)
	    : "r" (v), "r" (m), "n" (__ASI_P_L));
}

static inline void
__swapm64(volatile __uint64_t *m, __uint64_t v)
{
	__asm("stxa %1, [%2] %3 ! %0"
	    : "=m" (*m)
	    : "r" (v), "r" (m), "n" (__ASI_P_L));
}

#undef __ASI_P_L

#define __HAVE_MD_SWAPIO

#endif  /* _KERNEL */

#define __STRICT_ALIGNMENT

#ifndef __FROM_SYS__ENDIAN
#include <sys/endian.h>
#endif

#endif /* _MACHINE_ENDIAN_H_ */
