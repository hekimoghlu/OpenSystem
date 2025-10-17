/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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

#include <mach/mach_types.h>
#include <mach/kern_return.h>

#include <i386/lapic.h>
#include <i386/cpuid.h>
#include <i386/proc_reg.h>
#include <i386/machine_cpu.h>
#include <i386/misc_protos.h>
#include <i386/mp.h>
#include <i386/postcode.h>
#include <i386/cpu_threads.h>
#include <i386/machine_routines.h>
#include <i386/tsc.h>

#include <sys/kdebug.h>

/* Base vector for local APIC interrupt sources */
int lapic_interrupt_base = LAPIC_DEFAULT_INTERRUPT_BASE;

int             lapic_to_cpu[MAX_LAPICIDS];
int             cpu_to_lapic[MAX_CPUS];

void
lapic_cpu_map_init(void)
{
	int     i;

	for (i = 0; i < MAX_CPUS; i++) {
		cpu_to_lapic[i] = -1;
	}
	for (i = 0; i < MAX_LAPICIDS; i++) {
		lapic_to_cpu[i] = -1;
	}
}

void
lapic_cpu_map(int apic_id, int cpu)
{
	assert(apic_id < MAX_LAPICIDS);
	assert(cpu < MAX_CPUS);
	cpu_to_lapic[cpu] = apic_id;
	lapic_to_cpu[apic_id] = cpu;
}

/*
 * Retrieve the local apic ID a cpu.
 *
 * Returns the local apic ID for the given processor.
 * If the processor does not exist or apic not configured, returns -1.
 */

uint32_t
ml_get_apicid(uint32_t cpu)
{
	if (cpu >= (uint32_t)MAX_CPUS) {
		return 0xFFFFFFFF;      /* Return -1 if cpu too big */
	}
	/* Return the apic ID (or -1 if not configured) */
	return (uint32_t)cpu_to_lapic[cpu];
}

uint32_t
ml_get_cpuid(uint32_t lapic_index)
{
	if (lapic_index >= (uint32_t)MAX_LAPICIDS) {
		return 0xFFFFFFFF;      /* Return -1 if cpu too big */
	}
	/* Return the cpu ID (or -1 if not configured) */
	return (uint32_t)lapic_to_cpu[lapic_index];
}
