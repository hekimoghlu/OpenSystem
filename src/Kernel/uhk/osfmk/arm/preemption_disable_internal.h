/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#include <kern/percpu.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#pragma once

#if CONFIG_SPTM
/*
 * This header is only meant for the PPL and SPTM to access the preemption disable data structure directly.
 */
#endif /* CONFIG_SPTM */

/**
 * Track time and other counters during a preemption disabled window,
 * when `SCHED_HYGIENE` is configured.
 */
struct _preemption_disable_pcpu {
	/**
	 * A snapshot of times and counters relevant to preemption disable measurement.
	 */
	struct _preemption_disable_snap {
		/* The time when preemption was disabled, in Mach time units. */
		uint64_t pds_mach_time;
		/* The amount of time spent in interrupts by the current CPU, in Mach time units. */
		uint64_t pds_int_mach_time;
#if CONFIG_CPU_COUNTERS
		/* The number of cycles elapsed on this CPU. */
		uint64_t pds_cycles;
		/* The number of instructions seen by this CPU. */
		uint64_t pds_instrs;
#endif /* CONFIG_CPU_COUNTERS */
	}
	/* At the start of the preemption disabled window. */
	pdp_start;

	/* The maximum duration seen by this CPU, in Mach time units. */
	_Atomic uint64_t pdp_max_mach_duration;
	/*
	 * Whether to abandon the measurement on this CPU,
	 * due to a call to abandon_preemption_disable_measurement().
	 */
	bool pdp_abandon;
};

PERCPU_DECL(struct _preemption_disable_pcpu, _preemption_disable_pcpu_data);
