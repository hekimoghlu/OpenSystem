/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#if CONFIG_IOTRACE

#pragma once

#include <kern/percpu.h>
#include <libkern/OSDebug.h>
#include <stdint.h>

#define MAX_IOTRACE_BTFRAMES (16)

typedef enum {
	IOTRACE_PHYS_READ = 1,
	IOTRACE_PHYS_WRITE,
	IOTRACE_IO_READ,
	IOTRACE_IO_WRITE,
	IOTRACE_PORTIO_READ,
	IOTRACE_PORTIO_WRITE
} iotrace_type_e;

typedef struct {
	iotrace_type_e  iotype;
	int             size;
	uint64_t        vaddr;
	uint64_t        paddr;
	uint64_t        val;
	uint64_t        start_time_abs;
	uint64_t        duration;
	void           *backtrace[MAX_IOTRACE_BTFRAMES];
} iotrace_entry_t;

extern volatile int mmiotrace_enabled;
extern uint32_t iotrace_entries_per_cpu;

PERCPU_DECL(uint32_t, iotrace_next);
PERCPU_DECL(iotrace_entry_t * __unsafe_indexable, iotrace_ring);

static inline void
iotrace(iotrace_type_e type, uint64_t vaddr, uint64_t paddr, int size, uint64_t val,
    uint64_t sabs, uint64_t duration)
{
	uint32_t nextidx;
	iotrace_entry_t *cur_iotrace_ring;
	uint32_t *nextidxp;

	if (__improbable(mmiotrace_enabled == 0 ||
	    iotrace_entries_per_cpu == 0)) {
		return;
	}

	nextidxp = PERCPU_GET(iotrace_next);
	nextidx = *nextidxp;
	cur_iotrace_ring = *PERCPU_GET(iotrace_ring);

	cur_iotrace_ring[nextidx].iotype = type;
	cur_iotrace_ring[nextidx].vaddr = vaddr;
	cur_iotrace_ring[nextidx].paddr = paddr;
	cur_iotrace_ring[nextidx].size = size;
	cur_iotrace_ring[nextidx].val = val;
	cur_iotrace_ring[nextidx].start_time_abs = sabs;
	cur_iotrace_ring[nextidx].duration = duration;

	*nextidxp = ((nextidx + 1) >= iotrace_entries_per_cpu) ? 0 : (nextidx + 1);

	(void) OSBacktrace(cur_iotrace_ring[nextidx].backtrace,
	    MAX_IOTRACE_BTFRAMES);
}

static inline void
iotrace_disable(void)
{
	mmiotrace_enabled = 0;
}

#else /* CONFIG_IOTRACE */

#define iotrace_disable()
#define iotrace(type, vaddr, paddr, size, val, sabs, duration)

#endif /* CONFIG_IOTRACE */
