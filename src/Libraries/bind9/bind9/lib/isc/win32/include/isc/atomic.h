/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
/* $Id$ */

#ifndef ISC_ATOMIC_H
#define ISC_ATOMIC_H 1

#include <config.h>
#include <isc/platform.h>
#include <isc/types.h>

/*
 * This routine atomically increments the value stored in 'p' by 'val', and
 * returns the previous value.
 */
#ifdef ISC_PLATFORM_HAVEXADD
static __inline isc_int32_t
isc_atomic_xadd(isc_int32_t *p, isc_int32_t val) {
	return (isc_int32_t) _InterlockedExchangeAdd((long *)p, (long)val);
}
#endif

#ifdef ISC_PLATFORM_HAVEXADDQ
static __inline isc_int64_t
isc_atomic_xaddq(isc_int64_t *p, isc_int64_t val) {
	return (isc_int64_t) _InterlockedExchangeAdd64((__int64 *)p,
						       (__int64) val);
}
#endif

/*
 * This routine atomically stores the value 'val' in 'p' (32-bit version).
 */
#ifdef ISC_PLATFORM_HAVEATOMICSTORE
static __inline void
isc_atomic_store(isc_int32_t *p, isc_int32_t val) {
	(void) _InterlockedExchange((long *)p, (long)val);
}
#endif

/*
 * This routine atomically stores the value 'val' in 'p' (64-bit version).
 */
#ifdef ISC_PLATFORM_HAVEATOMICSTOREQ
static __inline void
isc_atomic_storeq(isc_int64_t *p, isc_int64_t val) {
	(void) _InterlockedExchange64((__int64 *)p, (__int64)val);
}
#endif

/*
 * This routine atomically replaces the value in 'p' with 'val', if the
 * original value is equal to 'cmpval'.  The original value is returned in any
 * case.
 */
#ifdef ISC_PLATFORM_HAVECMPXCHG
static __inline isc_int32_t
isc_atomic_cmpxchg(isc_int32_t *p, isc_int32_t cmpval, isc_int32_t val) {
	/* beware: swap arguments */
	return (isc_int32_t) _InterlockedCompareExchange((long *)p,
							 (long)val,
							 (long)cmpval);
}
#endif

#endif /* ISC_ATOMIC_H */
