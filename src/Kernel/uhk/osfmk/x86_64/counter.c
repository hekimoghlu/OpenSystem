/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#include <kern/assert.h>
#include <kern/cpu_data.h>
#include <kern/counter.h>
#include <kern/zalloc.h>
#include <machine/atomic.h>
#include <machine/machine_routines.h>
#include <machine/cpu_number.h>

OS_OVERLOADABLE
void
counter_add(scalable_counter_t *counter, uint64_t amount)
{
	disable_preemption();
	(*zpercpu_get(*counter)) += amount;
	enable_preemption();
}

OS_OVERLOADABLE
void
counter_inc(scalable_counter_t *counter)
{
	disable_preemption();
	(*zpercpu_get(*counter))++;
	enable_preemption();
}

OS_OVERLOADABLE
void
counter_dec(scalable_counter_t *counter)
{
	disable_preemption();
	(*zpercpu_get(*counter))--;
	enable_preemption();
}

OS_OVERLOADABLE
void
counter_add_preemption_disabled(scalable_counter_t *counter, uint64_t amount)
{
	(*zpercpu_get(*counter)) += amount;
}

OS_OVERLOADABLE
void
counter_inc_preemption_disabled(scalable_counter_t *counter)
{
	(*zpercpu_get(*counter))++;
}

OS_OVERLOADABLE
void
counter_dec_preemption_disabled(scalable_counter_t *counter)
{
	(*zpercpu_get(*counter))--;
}
