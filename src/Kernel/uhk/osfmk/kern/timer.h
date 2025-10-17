/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
/*
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */

#ifndef _KERN_TIMER_H_
#define _KERN_TIMER_H_

#include <kern/kern_types.h>

/*
 * Definitions for high resolution timers.
 */

#if __LP64__
#define TIMER_ALIGNMENT
#else
#define TIMER_ALIGNMENT __attribute__((packed, aligned(4)))
#endif

struct timer {
	uint64_t tstamp;
#if defined(__LP64__)
	uint64_t all_bits;
#else /* defined(__LP64__) */
	/* A check word on the high portion allows atomic updates. */
	uint32_t low_bits;
	uint32_t high_bits;
	uint32_t high_bits_check;
#endif /* !defined(__LP64__) */
} TIMER_ALIGNMENT;

typedef struct timer timer_data_t, *timer_t;

/*
 * Initialize the `timer`.
 */
void timer_init(timer_t timer);

/*
 * Start the `timer` at time `tstamp`.
 */
void timer_start(timer_t timer, uint64_t tstamp);

/*
 * Stop the `timer` and update it with time `tstamp`.
 */
void timer_stop(timer_t timer, uint64_t tstamp);

/*
 * Update the `timer` at time `tstamp`, leaving it running.
 */
void timer_update(timer_t timer, uint64_t tstamp);

/*
 * Read the accumulated time of `timer`.
 */
#if defined(__LP64__)
static inline
uint64_t
timer_grab(timer_t timer)
{
	return timer->all_bits;
}
#else /* defined(__LP64__) */
uint64_t timer_grab(timer_t timer);
#endif /* !defined(__LP64__) */

#endif /* _KERN_TIMER_H_ */
