/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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

// Definitions that are common to Nano V1 and Nano V2.
#if TARGET_OS_OSX || TARGET_OS_SIMULATOR || TARGET_OS_DRIVERKIT
#define NANO_PREALLOCATE_BAND_VM 0
#else // TARGET_OS_OSX || TARGET_OS_SIMULATOR || TARGET_OS_DRIVERKIT
#define NANO_PREALLOCATE_BAND_VM 1 // pre-allocate reserved vm range
#endif // TARGET_OS_OSX || TARGET_OS_SIMULATOR || TARGET_OS_DRIVERKIT

typedef enum {
	NANO_NONE	= 0,
	NANO_V1 	= 1,
	NANO_V2		= 2,
} nano_version_t;

// Nano malloc enabled flag
MALLOC_NOEXPORT
extern nano_version_t _malloc_engaged_nano;

// The maximum number of per-CPU allocation regions to use for Nano.
MALLOC_NOEXPORT
extern unsigned int nano_common_max_magazines;

MALLOC_NOEXPORT
extern boolean_t nano_common_max_magazines_is_ncpu;

MALLOC_NOEXPORT
void
nano_common_cpu_number_override_set(void);

MALLOC_NOEXPORT
void
nano_common_init(const char *envp[], const char *apple[], const char *bootargs);

MALLOC_NOEXPORT
void
nano_common_configure(void);

MALLOC_NOEXPORT
void *
nano_common_allocate_based_pages(size_t size, unsigned char align,
		unsigned debug_flags, int vm_page_label, void *base_addr);

MALLOC_NOEXPORT
boolean_t
nano_common_allocate_vm_space(mach_vm_address_t base, mach_vm_size_t size);

MALLOC_NOEXPORT
void
nano_common_deallocate_pages(void *addr, size_t size, unsigned debug_flags);

MALLOC_NOEXPORT
kern_return_t
nano_common_default_reader(task_t task, vm_address_t address, vm_size_t size,
		void **ptr);

#endif // __NANO_MALLOC_COMMON_H

