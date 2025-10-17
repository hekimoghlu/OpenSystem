/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

ZONE_VIEW_DEFINE(counters_zone, "per_cpu_counters", .zv_zone = &percpu_u64_zone, sizeof(uint64_t));

/*
 * Tracks how many static scalable counters are in use since they won't show up
 * in the per_cpu_counters zone stats.
 */
uint64_t num_static_scalable_counters;

/*
 * Mangle the given scalable_counter_t so that it points to the early storage
 * regardless of which CPU # we're boot on.
 * Must be run before we go multi-core.
 */
__startup_func void
scalable_counter_static_boot_mangle(scalable_counter_t *counter)
{
	*counter = __zpcpu_mangle_for_boot(*counter);
}

/*
 * Initializes a static counter in permanent per-cpu memory.
 * Run during startup for each static per-cpu counter
 * Must be run before we go multi-core.
 */
__startup_func void
scalable_counter_static_init(scalable_counter_t *counter)
{
	/*
	 * We pointed the counter to a single global value during early boot.
	 * Grab that value now. We'll store it in our current CPU's value
	 */
	uint64_t current_value = os_atomic_load_wide(zpercpu_get(*counter), relaxed);
	/*
	 * This counter can't be freed so we allocate it out of the permanent zone rather than
	 * our counter zone.
	 */
	*counter = zalloc_percpu_permanent(sizeof(uint64_t), ZALIGN_64);
	os_atomic_store_wide(zpercpu_get(*counter), current_value, relaxed);
	num_static_scalable_counters++;
}

OS_OVERLOADABLE
void
counter_alloc(scalable_counter_t *counter)
{
	*counter = zalloc_percpu(counters_zone, Z_WAITOK | Z_ZERO | Z_NOFAIL);
}

OS_OVERLOADABLE
void
counter_alloc(atomic_counter_t *counter)
{
	os_atomic_store_wide(counter, 0, relaxed);
}

OS_OVERLOADABLE
void
counter_free(scalable_counter_t *counter)
{
	zfree_percpu(counters_zone, *counter);
}

OS_OVERLOADABLE
void
counter_free(atomic_counter_t *counter)
{
	(void)counter;
}

OS_OVERLOADABLE
void
counter_add(atomic_counter_t *counter, uint64_t amount)
{
	os_atomic_add(counter, amount, relaxed);
}

OS_OVERLOADABLE
void
counter_inc(atomic_counter_t *counter)
{
	os_atomic_inc(counter, relaxed);
}

OS_OVERLOADABLE
void
counter_dec(atomic_counter_t *counter)
{
	os_atomic_dec(counter, relaxed);
}

OS_OVERLOADABLE
void
counter_add_preemption_disabled(atomic_counter_t *counter, uint64_t amount)
{
	counter_add(counter, amount);
}

OS_OVERLOADABLE
void
counter_inc_preemption_disabled(atomic_counter_t *counter)
{
	counter_inc(counter);
}

OS_OVERLOADABLE
void
counter_dec_preemption_disabled(atomic_counter_t *counter)
{
	counter_dec(counter);
}

OS_OVERLOADABLE
uint64_t
counter_load(atomic_counter_t *counter)
{
	return os_atomic_load_wide(counter, relaxed);
}

OS_OVERLOADABLE
uint64_t
counter_load(scalable_counter_t *counter)
{
	uint64_t value = 0;
	zpercpu_foreach(it, *counter) {
		value += os_atomic_load_wide(it, relaxed);
	}
	return value;
}
