/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#include <arm/pmap/pmap_internal.h>
#include <arm/preemption_disable_internal.h>

/**
 * Placeholder for random pmap functionality that doesn't fit into any of the
 * other files. This will contain things like the CPU copy windows, ASID
 * management and context switching code, and stage 2 pmaps (among others).
 *
 * My idea is that code that doesn't fit into any of the other files will live
 * in this file until we deem it large and important enough to break into its
 * own file.
 */


void
pmap_abandon_measurement(void)
{
#if SCHED_HYGIENE_DEBUG
	struct _preemption_disable_pcpu *pcpu = PERCPU_GET(_preemption_disable_pcpu_data);
	uint64_t istate = pmap_interrupts_disable();
	if (pcpu->pdp_start.pds_mach_time != 0) {
		pcpu->pdp_abandon = true;
	}
	pmap_interrupts_restore(istate);
#endif /* SCHED_HYGIENE_DEBUG */
}
