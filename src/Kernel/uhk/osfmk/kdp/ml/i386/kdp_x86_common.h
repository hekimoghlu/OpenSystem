/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#ifndef _KDP_X86_COMMON_H_
#define _KDP_X86_COMMON_H_

#include <libsa/types.h>
#include <mach/machine/vm_types.h>
#include <i386/pmap.h>

/* data required for JTAG extraction of coredump */
typedef struct _kdp_jtag_coredump_t {
	uint64_t signature;
	uint64_t version;
	uint64_t kernel_map_start;
	uint64_t kernel_map_end;
	uint64_t kernel_pmap_pml4;
	uint64_t pmap_memory_regions;
	uint64_t pmap_memory_region_count;
	uint64_t pmap_memory_region_t_size;
	uint64_t physmap_base;
} kdp_jtag_coredump_t;

/* signature used to verify kdp_jtag_coredump_t structure */
#define KDP_JTAG_COREDUMP_SIGNATURE     0x434f524544554d50

/* version of kdp_jtag_coredump_t structure */
#define KDP_JTAG_COREDUMP_VERSION_1     1

void kdp_map_debug_pagetable_window(void);
void kdp_jtag_coredump_init(void);

#endif /* _KDP_X86_COMMON_H_ */
