/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#define __APPLE_API_PRIVATE 1
#define __APPLE_API_UNSTABLE 1

#include <kern/debug.h>
#include <kern/iotrace.h>
#include <kern/zalloc.h>

#include <pexpert/pexpert.h>

#define DEFAULT_IOTRACE_ENTRIES_PER_CPU (186)
#define IOTRACE_MAX_ENTRIES_PER_CPU (1024)

volatile int mmiotrace_enabled = 1;
uint32_t iotrace_entries_per_cpu;

uint32_t PERCPU_DATA(iotrace_next);
iotrace_entry_t *PERCPU_DATA(iotrace_ring);

static void
init_iotrace_bufs(int entries_per_cpu)
{
	const size_t size = entries_per_cpu * sizeof(iotrace_entry_t);

	percpu_foreach(ring, iotrace_ring) {
		*ring = zalloc_permanent_tag(size, ZALIGN(iotrace_entry_t),
		    VM_KERN_MEMORY_DIAG);
	};

	iotrace_entries_per_cpu = entries_per_cpu;
}

__startup_func
static void
iotrace_init(void)
{
	int entries_per_cpu = DEFAULT_IOTRACE_ENTRIES_PER_CPU;
	int enable = mmiotrace_enabled;

	if (kern_feature_override(KF_IOTRACE_OVRD)) {
		enable = 0;
	}

	(void) PE_parse_boot_argn("iotrace", &enable, sizeof(enable));
	if (enable != 0 &&
	    PE_parse_boot_argn("iotrace_epc", &entries_per_cpu, sizeof(entries_per_cpu)) &&
	    (entries_per_cpu < 1 || entries_per_cpu > IOTRACE_MAX_ENTRIES_PER_CPU)) {
		entries_per_cpu = DEFAULT_IOTRACE_ENTRIES_PER_CPU;
	}

	mmiotrace_enabled = enable;

	if (mmiotrace_enabled) {
		init_iotrace_bufs(entries_per_cpu);
	}
}

STARTUP(EARLY_BOOT, STARTUP_RANK_MIDDLE, iotrace_init);
