/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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

#include <mach/kern_return.h>
#include <mach/port.h>
#include <kern/queue.h>
#include <kern/processor.h>
#include <kern/thread.h>
#include <kern/sched_prim.h>
#include <kern/timer.h>

#include <machine/config.h>

void
timer_init(timer_t timer)
{
	memset(timer, 0, sizeof(*timer));
}

static void
timer_advance(timer_t timer, uint64_t delta)
{
#if defined(__LP64__)
	timer->all_bits += delta;
#else /* defined(__LP64__) */
	extern void timer_advance_internal_32(timer_t timer, uint32_t high,
	    uint32_t low);
	uint64_t low = delta + timer->low_bits;
	if (low >> 32) {
		timer_advance_internal_32(timer,
		    (uint32_t)(timer->high_bits + (low >> 32)), (uint32_t)low);
	} else {
		timer->low_bits = (uint32_t)low;
	}
#endif /* defined(__LP64__) */
}

void
timer_start(timer_t timer, uint64_t tstamp)
{
	timer->tstamp = tstamp;
}

void
timer_stop(timer_t timer, uint64_t tstamp)
{
	timer_advance(timer, tstamp - timer->tstamp);
}

void
timer_update(timer_t timer, uint64_t tstamp)
{
	timer_advance(timer, tstamp - timer->tstamp);
	timer->tstamp = tstamp;
}
