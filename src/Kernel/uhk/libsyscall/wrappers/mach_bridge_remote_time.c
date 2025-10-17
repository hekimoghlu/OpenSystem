/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#include <sys/types.h>
#include <machine/cpu_capabilities.h>
#include <kern/remote_time.h>
#include <mach/mach_time.h>
#include "strings.h"
#include <TargetConditionals.h>

#define BT_RESET_SENTINEL_TS  (~3ULL) /* from machine/machine_remote_time.h */

extern uint64_t __mach_bridge_remote_time(uint64_t local_time);

#if TARGET_OS_BRIDGE && defined(__arm64__)
static uint64_t
absolutetime_to_nanoseconds(uint64_t abs_time)
{
	mach_timebase_info_data_t info;
	mach_timebase_info(&info);
	uint64_t time_in_ns = (uint64_t)(((double)info.numer / (double)info.denom) * abs_time);

	return time_in_ns;
}

uint64_t
mach_bridge_remote_time(__unused uint64_t local_time)
{
	uint64_t remote_time = 0;
	uint64_t local_time_ns = 0;
	uint64_t now = 0;
	struct bt_params params = {};

	COMM_PAGE_SLOT_TYPE(struct bt_params) commpage_bt_params_p =
	    COMM_PAGE_SLOT(struct bt_params, REMOTETIME_PARAMS);
	volatile uint64_t *base_local_ts_p = &commpage_bt_params_p->base_local_ts;
	volatile uint64_t *base_remote_ts_p = &commpage_bt_params_p->base_remote_ts;
	volatile double *rate_p = &commpage_bt_params_p->rate;

	do {
		params.base_local_ts = *base_local_ts_p;
		if (*base_local_ts_p == BT_RESET_SENTINEL_TS) {
			return 0;
		}
		/*
		 * This call contains an instruction barrier that ensures the second read of
		 * base_local_ts is not speculated above the first read of base_local_ts.
		 */
		now = mach_absolute_time();
		params.base_remote_ts = *base_remote_ts_p;
		params.rate = *rate_p;
		/*
		 * This barrier prevents the second read of base_local_ts from being reordered
		 * w.r.t the reads of other values in bt_params.
		 */
		__asm__ volatile ("dmb ishld" ::: "memory");
	} while (params.base_local_ts && (params.base_local_ts != commpage_bt_params_p->base_local_ts));

	if (!local_time) {
		local_time = now;
	}
	local_time_ns = absolutetime_to_nanoseconds(local_time);
	if (local_time_ns < params.base_local_ts) {
		remote_time = __mach_bridge_remote_time(local_time);
	} else {
		remote_time = mach_bridge_compute_timestamp(local_time_ns, &params);
	}
	return remote_time;
}
#endif /* TARGET_OS_BRIDGE && defined(__arm64__) */
