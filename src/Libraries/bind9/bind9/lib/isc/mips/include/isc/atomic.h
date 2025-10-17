/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
/* $Id: atomic.h,v 1.3 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_ATOMIC_H
#define ISC_ATOMIC_H 1

#include <isc/platform.h>
#include <isc/types.h>

#ifdef ISC_PLATFORM_USEGCCASM
/*
 * This routine atomically increments the value stored in 'p' by 'val', and
 * returns the previous value.
 */
static inline isc_int32_t
isc_atomic_xadd(isc_int32_t *p, int val) {
	isc_int32_t orig;

	__asm__ __volatile__ (
	"	.set	push		\n"
	"	.set	mips2		\n"
	"	.set	noreorder	\n"
	"	.set	noat		\n"
	"1:	ll	$1, %1		\n"
	"	addu	%0, $1, %2	\n"
	"	sc	%0, %1		\n"
	"	beqz	%0, 1b		\n"
	"	move	%0, $1		\n"
	"	.set	pop		\n"
	: "=&r" (orig), "+R" (*p)
	: "r" (val)
	: "memory");

	return (orig);
}

/*
 * This routine atomically stores the value 'val' in 'p'.
 */
static inline void
isc_atomic_store(isc_int32_t *p, isc_int32_t val) {
	*p = val;
}

/*
 * This routine atomically replaces the value in 'p' with 'val', if the
 * original value is equal to 'cmpval'.  The original value is returned in any
 * case.
 */
static inline isc_int32_t
isc_atomic_cmpxchg(isc_int32_t *p, int cmpval, int val) {
	isc_int32_t orig;
	isc_int32_t tmp;

	__asm__ __volatile__ (
	"	.set	push		\n"
	"	.set	mips2		\n"
	"	.set	noreorder	\n"
	"	.set	noat		\n"
	"1:	ll	$1, %1		\n"
	"	bne	$1, %3, 2f	\n"
	"	move	%2, %4		\n"
	"	sc	%2, %1		\n"
	"	beqz	%2, 1b		\n"
	"2:	move	%0, $1		\n"
	"	.set	pop		\n"
	: "=&r"(orig), "+R" (*p), "=r" (tmp)
	: "r"(cmpval), "r"(val)
	: "memory");

	return (orig);
}

#else /* !ISC_PLATFORM_USEGCCASM */

#error "unsupported compiler.  disable atomic ops by --disable-atomic"

#endif
#endif /* ISC_ATOMIC_H */
