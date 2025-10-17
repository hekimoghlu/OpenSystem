/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
/* Public Domain */

#ifndef _MACHINE_ATOMIC_H_
#define _MACHINE_ATOMIC_H_

#define __membar(_f) do { __asm volatile(_f ::: "memory"); } while (0)

#define membar_enter()		__membar("dmb sy")
#define membar_exit()		__membar("dmb sy")
#define membar_producer()	__membar("dmb st")
#define membar_consumer()	__membar("dmb ld")
#define membar_sync()		__membar("dmb sy")

#if defined(_KERNEL)

/* virtio needs MP membars even on SP kernels */
#define virtio_membar_producer()	__membar("dmb st")
#define virtio_membar_consumer()	__membar("dmb ld")
#define virtio_membar_sync()		__membar("dmb sy")

/*
 * Set bits
 * *p = *p | v
 */
static inline void
atomic_setbits_int(volatile unsigned int *p, unsigned int v)
{
	unsigned int modified, tmp;

	__asm volatile (
	    "1:	ldxr %w0, [%x3]		\n\t"
	    "	orr %w0, %w0, %w2	\n\t"
	    "	stxr %w1, %w0, [%x3]	\n\t"
	    "	cbnz %w1, 1b		\n\t"
	    : "=&r" (tmp), "=&r" (modified)
	    : "r" (v), "r" (p)
	    : "memory", "cc"
	);
}

/*
 * Clear bits
 * *p = *p & (~v)
 */
static inline void
atomic_clearbits_int(volatile unsigned int *p, unsigned int v)
{
	unsigned int modified, tmp;

	__asm volatile (
	    "1:	ldxr %w0, [%x3]		\n\t"
	    "	bic %w0, %w0, %w2	\n\t"
	    "	stxr %w1, %w0, [%x3]	\n\t"
	    "	cbnz %w1, 1b		\n\t"
	    : "=&r" (tmp), "=&r" (modified)
	    : "r" (v), "r" (p)
	    : "memory", "cc"
	);
}

#endif /* defined(_KERNEL) */
#endif /* _MACHINE_ATOMIC_H_ */
