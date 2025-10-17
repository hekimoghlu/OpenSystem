/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#include <machine/atomic.h>
#include <mach/mach_time.h>
#include <mach/clock_types.h>
#include <kern/clock.h>
#include <kern/locks.h>
#include <arm64/machine_remote_time.h>
#include <sys/kdebug.h>
#include <arm/machine_routines.h>
#include <kern/remote_time.h>

_Atomic uint32_t bt_init_flag = 0;

extern void mach_bridge_add_timestamp(uint64_t remote_timestamp, uint64_t local_timestamp);
extern void bt_calibration_thread_start(void);
extern void bt_params_add(struct bt_params *params);

void
mach_bridge_init_timestamp(void)
{
	/* This function should be called only once by the driver
	 *  implementing the interrupt handler for receiving timestamps */
	if (os_atomic_load(&bt_init_flag, relaxed)) {
		return;
	}

	os_atomic_store(&bt_init_flag, 1, release);

	/* Start the kernel thread only after all the locks have been initialized */
	bt_calibration_thread_start();
}

/*
 * Conditions: Should be called from primary interrupt context
 */
void
mach_bridge_recv_timestamps(uint64_t remoteTimestamp, uint64_t localTimestamp)
{
	assert(ml_at_interrupt_context() == TRUE);

	/* Ensure the locks have been initialized */
	if (!os_atomic_load(&bt_init_flag, acquire)) {
		panic("%s called before mach_bridge_init_timestamp", __func__);
		return;
	}

	KDBG(MACHDBG_CODE(DBG_MACH_CLOCK, MACH_BRIDGE_RCV_TS), localTimestamp, remoteTimestamp);

	lck_spin_lock(&bt_spin_lock);
	mach_bridge_add_timestamp(remoteTimestamp, localTimestamp);
	lck_spin_unlock(&bt_spin_lock);

	return;
}

/*
 * This function is used to set parameters, calculated externally,
 * needed for mach_bridge_remote_time.
 */
void
mach_bridge_set_params(uint64_t local_timestamp, uint64_t remote_timestamp, double rate)
{
	/* Ensure the locks have been initialized */
	if (!os_atomic_load(&bt_init_flag, acquire)) {
		panic("%s called before mach_bridge_init_timestamp", __func__);
		return;
	}

	struct bt_params params = {};
	params.base_local_ts = local_timestamp;
	params.base_remote_ts = remote_timestamp;
	params.rate = rate;
	lck_spin_lock(&bt_ts_conversion_lock);
	bt_params_add(&params);
	lck_spin_unlock(&bt_ts_conversion_lock);
	KDBG(MACHDBG_CODE(DBG_MACH_CLOCK, MACH_BRIDGE_TS_PARAMS), params.base_local_ts,
	    params.base_remote_ts, *(uint64_t *)((void *)&params.rate));
}
