/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include <mach/mach_types.h>
#include <kern/task.h> /* task_ledgers */
#include <kern/ledger.h>

#include <kperf/kperf.h>

#include <kperf/buffer.h>
#include <kperf/context.h>
#include <kperf/meminfo.h>

/* collect current memory info */
void
kperf_meminfo_sample(task_t task, struct meminfo *mi)
{
	ledger_amount_t credit, debit;
	kern_return_t kr;

	assert(mi != NULL);

	BUF_INFO(PERF_MI_SAMPLE | DBG_FUNC_START);

	mi->phys_footprint = get_task_phys_footprint(task);

	kr = ledger_get_entries(task->ledger, task_ledgers.purgeable_volatile,
	    &credit, &debit);
	if (kr == KERN_SUCCESS) {
		mi->purgeable_volatile = credit - debit;
	} else {
		mi->purgeable_volatile = UINT64_MAX;
	}

	kr = ledger_get_entries(task->ledger,
	    task_ledgers.purgeable_volatile_compressed,
	    &credit, &debit);
	if (kr == KERN_SUCCESS) {
		mi->purgeable_volatile_compressed = credit - debit;
	} else {
		mi->purgeable_volatile_compressed = UINT64_MAX;
	}

	BUF_INFO(PERF_MI_SAMPLE | DBG_FUNC_END);
}

/* log an existing sample into the buffer */
void
kperf_meminfo_log(struct meminfo *mi)
{
	BUF_DATA(PERF_MI_DATA, mi->phys_footprint, mi->purgeable_volatile,
	    mi->purgeable_volatile_compressed);
}
