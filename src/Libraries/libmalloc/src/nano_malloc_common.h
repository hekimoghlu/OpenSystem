/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#ifndef __NANO_MALLOC_COMMON_H
#define __NANO_MALLOC_COMMON_H

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

typedef enum {
	NANO_NONE	= 0,
	NANO_V2		= 2,
} nano_version_t;

// Nano malloc enabled flag
//
// Note that this flag indicates whether a "nano-like" configuration, i.e. one
// that trades memory for CPU and scalability, is enabled, but not literally
// whether the nano allocator itself is engaged.  Test for nano itself using
// initial_nano_zone.
MALLOC_NOEXPORT
extern nano_version_t _malloc_engaged_nano;

// The maximum number of per-CPU allocation regions to use for Nano.
MALLOC_NOEXPORT
extern unsigned int nano_common_max_magazines;

MALLOC_NOEXPORT
extern bool nano_common_max_magazines_is_ncpu;

// Index of last region to be allocated
MALLOC_NOEXPORT
extern unsigned int nano_max_region;

MALLOC_NOEXPORT
void
nano_common_cpu_number_override_set(void);

MALLOC_NOEXPORT
void
nano_common_init(const char * __null_terminated * __null_terminated envp, const char * __null_terminated * __null_terminated apple, const char *bootargs);

MALLOC_NOEXPORT
void
nano_common_configure(void);

MALLOC_NOEXPORT
void *
nano_common_allocate_based_pages(size_t size, unsigned char align,
		unsigned debug_flags, int vm_page_label, void *base_addr);

MALLOC_NOEXPORT
bool
nano_common_allocate_vm_space(mach_vm_address_t base, mach_vm_size_t size);

MALLOC_NOEXPORT
bool
nano_common_reserve_vm_space(mach_vm_address_t base, mach_vm_size_t size);

MALLOC_NOEXPORT
bool
nano_common_unprotect_vm_space(mach_vm_address_t base, mach_vm_size_t size);

MALLOC_NOEXPORT
void
nano_common_deallocate_pages(void *addr, size_t size, unsigned debug_flags);

MALLOC_NOEXPORT
kern_return_t
nano_common_default_reader(task_t task, vm_address_t address, vm_size_t size,
		void **ptr);

MALLOC_NOEXPORT
nano_version_t
_nano_common_init_pick_mode(const char * __null_terminated * __null_terminated envp, const char * __null_terminated * __null_terminated apple, const char *bootargs, bool space_efficient_enabled);

#endif // __NANO_MALLOC_COMMON_H

