/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#ifndef _SYS_STACKSHOT_H
#define _SYS_STACKSHOT_H

#include <stdint.h>

#define STACKSHOT_CONFIG_TYPE 1

typedef struct stackshot_config {
	/* Input options */
	int             sc_pid;                 /* PID to trace, or -1 for the entire system */
	uint64_t        sc_flags;               /* Stackshot flags */
	uint64_t        sc_delta_timestamp;     /* Retrieve a delta stackshot of system state that has changed since this time */

	/* Stackshot results */
	uint64_t        sc_buffer;              /* Pointer to stackshot buffer */
	uint32_t        sc_size;                /* Length of the stackshot buffer */

	/* Internals */
	uint64_t        sc_out_buffer_addr;     /* Location where the kernel should copy the address of the newly mapped buffer in user space */
	uint64_t        sc_out_size_addr;       /* Location where the kernel should copy the size of the stackshot buffer */
	uint64_t                sc_pagetable_mask;      /* Mask of page table levels to dump, must pass STACKSHOT_PAGE_TABLES */
} stackshot_config_t;

typedef struct stackshot_stats {
	uint64_t        ss_last_start;          /* mach_absolute_time of last start */
	uint64_t        ss_last_end;            /* mach_absolute_time of last end */
	uint64_t        ss_count;               /* count of stackshots taken */
	uint64_t        ss_duration;            /* sum(mach_absolute_time) of taken stackshots */
} stackshot_stats_t;

#ifndef KERNEL

stackshot_config_t * stackshot_config_create(void);
int stackshot_config_set_pid(stackshot_config_t * stackshot_config, int pid);
int stackshot_config_set_flags(stackshot_config_t * stackshot_config, uint64_t flags);
int stackshot_capture_with_config(stackshot_config_t * stackshot_config);
void * stackshot_config_get_stackshot_buffer(stackshot_config_t * stackshot_config);
uint32_t stackshot_config_get_stackshot_size(stackshot_config_t * stackshot_config);
int stackshot_config_set_size_hint(stackshot_config_t * stackshot_config, uint32_t suggested_size);
int stackshot_config_set_delta_timestamp(stackshot_config_t * stackshot_config, uint64_t delta_timestamp);
int stackshot_config_set_pagetable_mask(stackshot_config_t * stackshot_config, uint32_t mask);
int stackshot_config_dealloc_buffer(stackshot_config_t * stackshot_config);
int stackshot_config_dealloc(stackshot_config_t * stackshot_config);

#endif /* ! KERNEL */

#endif /* _SYS_STACKSHOT_H */
