/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
	os_atomic_add(zpercpu_get(*counter), amount, relaxed);
}

OS_OVERLOADABLE
void
counter_inc(scalable_counter_t *counter)
{
	os_atomic_inc(zpercpu_get(*counter), relaxed);
}

OS_OVERLOADABLE
void
counter_dec(scalable_counter_t *counter)
{
	os_atomic_dec(zpercpu_get(*counter), relaxed);
}

/*
 * NB: On arm, the preemption disabled implementation is the same as
 * the normal implementation. Otherwise we would need to enforce that
 * callers never mix the interfaces for the same counter.
 */
OS_OVERLOADABLE
void
counter_add_preemption_disabled(scalable_counter_t *counter, uint64_t amount)
{
	counter_add(counter, amount);
}

OS_OVERLOADABLE
void
counter_inc_preemption_disabled(scalable_counter_t *counter)
{
	counter_inc(counter);
}

OS_OVERLOADABLE
void
counter_dec_preemption_disabled(scalable_counter_t *counter)
{
	counter_dec(counter);
}
