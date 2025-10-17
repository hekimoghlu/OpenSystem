/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#ifndef REMOTE_TIME_H
#define REMOTE_TIME_H

#include <sys/cdefs.h>
#include <stdint.h>
#include <os/overflow.h>

__BEGIN_DECLS
/* bt_params is an ABI for tracing tools */
struct bt_params {
	double rate;
	uint64_t base_local_ts;
	uint64_t base_remote_ts;
};

/* local_ts_ns should be in nanoseconds */
static inline uint64_t
mach_bridge_compute_timestamp(uint64_t local_ts_ns, struct bt_params *params)
{
	if (!params || params->rate == 0.0) {
		return 0;
	}
	/*
	 * Formula to compute remote_timestamp
	 * remote_timestamp = (bt_params.rate * (local_ts_ns - bt_params.base_local_ts))
	 *	 +  bt_params.base_remote_ts
	 */
	int64_t remote_ts = 0;
	int64_t rate_prod = 0;
	/* To avoid precision loss due to typecasting from int64_t to double */
	if (params->rate != 1.0) {
		rate_prod = (int64_t)(params->rate * (double)((int64_t)local_ts_ns - (int64_t)params->base_local_ts));
	} else {
		rate_prod = (int64_t)local_ts_ns - (int64_t)params->base_local_ts;
	}
	if (os_add_overflow((int64_t)params->base_remote_ts, rate_prod, &remote_ts)) {
		return 0;
	}

	return (uint64_t)remote_ts;
}

uint64_t mach_bridge_remote_time(uint64_t);

#if XNU_KERNEL_PRIVATE
#include <kern/locks.h>
extern lck_spin_t bt_maintenance_lock;
extern lck_spin_t bt_spin_lock;
extern lck_spin_t bt_ts_conversion_lock;
#endif

__END_DECLS

#endif /* REMOTE_TIME_H */
