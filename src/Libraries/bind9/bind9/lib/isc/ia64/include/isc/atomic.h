/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
/* $Id: atomic.h,v 1.7 2009/06/24 02:22:50 marka Exp $ */

#ifndef ISC_ATOMIC_H
#define ISC_ATOMIC_H 1

#include <isc/platform.h>
#include <isc/types.h>

#ifdef ISC_PLATFORM_USEGCCASM
/*
 * This routine atomically increments the value stored in 'p' by 'val', and
 * returns the previous value.
 *
 * Open issue: can 'fetchadd' make the code faster for some particular values
 * (e.g., 1 and -1)?
 */
static inline isc_int32_t
#ifdef __GNUC__
__attribute__ ((unused))
#endif
isc_atomic_xadd(isc_int32_t *p, isc_int32_t val)
{
	isc_int32_t prev, swapped;

	for (prev = *(volatile isc_int32_t *)p; ; prev = swapped) {
		swapped = prev + val;
		__asm__ volatile(
			"mov ar.ccv=%2;;"
			"cmpxchg4.acq %0=%4,%3,ar.ccv"
			: "=r" (swapped), "=m" (*p)
			: "r" (prev), "r" (swapped), "m" (*p)
			: "memory");
		if (swapped == prev)
			break;
	}

	return (prev);
}

/*
 * This routine atomically stores the value 'val' in 'p'.
 */
static inline void
#ifdef __GNUC__
__attribute__ ((unused))
#endif
isc_atomic_store(isc_int32_t *p, isc_int32_t val)
{
	__asm__ volatile(
		"st4.rel %0=%1"
		: "=m" (*p)
		: "r" (val)
		: "memory"
		);
}

/*
 * This routine atomically replaces the value in 'p' with 'val', if the
 * original value is equal to 'cmpval'.  The original value is returned in any
 * case.
 */
static inline isc_int32_t
#ifdef __GNUC__
__attribute__ ((unused))
#endif
isc_atomic_cmpxchg(isc_int32_t *p, isc_int32_t cmpval, isc_int32_t val)
{
	isc_int32_t ret;

	__asm__ volatile(
		"mov ar.ccv=%2;;"
		"cmpxchg4.acq %0=%4,%3,ar.ccv"
		: "=r" (ret), "=m" (*p)
		: "r" (cmpval), "r" (val), "m" (*p)
		: "memory");

	return (ret);
}
#else /* !ISC_PLATFORM_USEGCCASM */

#error "unsupported compiler.  disable atomic ops by --disable-atomic"

#endif
#endif /* ISC_ATOMIC_H */
