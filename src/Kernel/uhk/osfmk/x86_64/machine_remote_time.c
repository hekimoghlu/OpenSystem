/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include <kern/misc_protos.h>
#include <x86_64/machine_remote_time.h>
#include <machine/atomic.h>
#include <kern/locks.h>
#include <kern/clock.h>
#include <kern/remote_time.h>

void mach_bridge_send_timestamp(uint64_t timestamp);

extern _Atomic uint32_t bt_init_flag;
extern uint32_t bt_enable_flag;

/*
 * Delay sending timestamps by certain interval to
 * avoid overwriting sentinel values
 */
#define DELAY_INTERVAL_NS (50 * NSEC_PER_MSEC)
static uint64_t bt_delay_timestamp = 0;
static mach_bridge_regwrite_timestamp_func_t bridge_regwrite_timestamp_callback = NULL;

/*
 * This function should only be called by the kext
 * responsible for sending timestamps across the link
 */
void
mach_bridge_register_regwrite_timestamp_callback(mach_bridge_regwrite_timestamp_func_t func)
{
	static uint64_t delay_amount = 0;

	if (!os_atomic_load(&bt_init_flag, relaxed)) {
		nanoseconds_to_absolutetime(DELAY_INTERVAL_NS, &delay_amount);
		os_atomic_store(&bt_init_flag, 1, release);
	}

	lck_spin_lock(&bt_maintenance_lock);
	bridge_regwrite_timestamp_callback = func;
	bt_enable_flag = (func != NULL) ? 1 : 0;
	bt_delay_timestamp = mach_absolute_time() + delay_amount;
	lck_spin_unlock(&bt_maintenance_lock);
}

void
mach_bridge_send_timestamp(uint64_t timestamp)
{
	LCK_SPIN_ASSERT(&bt_maintenance_lock, LCK_ASSERT_OWNED);

	if (bt_delay_timestamp > 0) {
		uint64_t now = mach_absolute_time();
		if (now < bt_delay_timestamp) {
			return;
		}
		bt_delay_timestamp = 0;
	}

	if (bridge_regwrite_timestamp_callback) {
		bridge_regwrite_timestamp_callback(timestamp);
	}
}
