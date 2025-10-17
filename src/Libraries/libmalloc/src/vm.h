/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#ifndef __VM_H
#define __VM_H

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

#if MALLOC_TARGET_EXCLAVES
typedef struct _liblibc_plat_mem_map_t plat_map_t;

#define mvm_plat_map(x) (&(x))
#else
#define mvm_plat_map(x) NULL
#endif // MALLOC_TARGET_EXCLAVES

extern uint64_t malloc_entropy[2];

#if !MALLOC_TARGET_EXCLAVES
static inline bool
mvm_aslr_enabled(void)
{
	extern struct mach_header __dso_handle;
	return _dyld_get_image_slide(&__dso_handle);
}
#endif // !MALLOC_TARGET_EXCLAVES

MALLOC_NOEXPORT
void
mvm_aslr_init(void);

MALLOC_NOEXPORT
void * __alloc_size(2) __sized_by_or_null(size)
mvm_allocate_plat(uintptr_t addr, size_t size, uint8_t align, int flags, int debug_flags, int label, plat_map_t *map_out);

MALLOC_NOEXPORT
void * __alloc_size(1) __sized_by_or_null(size)
mvm_allocate_pages(size_t size, uint8_t align, uint32_t debug_flags, int vm_page_label);

MALLOC_NOEXPORT
void * __alloc_size(1) __sized_by_or_null(size)
mvm_allocate_pages_plat(size_t size, uint8_t align, uint32_t debug_flags, int vm_page_label, plat_map_t *map_out);

MALLOC_NOEXPORT
void
mvm_deallocate_plat(void * __sized_by(size) addr, size_t size, int debug_flags, plat_map_t *map);

MALLOC_NOEXPORT
void
mvm_deallocate_pages(void * __sized_by(size) addr, size_t size, unsigned debug_flags);

MALLOC_NOEXPORT
void
mvm_deallocate_pages_plat(void * __sized_by(size) addr, size_t size, unsigned debug_flags, plat_map_t *map);

MALLOC_NOEXPORT
int
mvm_madvise(void * __sized_by(size) addr, size_t size, int advice, unsigned debug_flags);

MALLOC_NOEXPORT
int
mvm_madvise_plat(void * __sized_by(size) addr, size_t size, int advice, unsigned debug_flags, plat_map_t *map);

MALLOC_NOEXPORT
int
mvm_madvise_free(void *szone, void *r, uintptr_t pgLo, uintptr_t pgHi, uintptr_t *last, boolean_t scribble);

MALLOC_NOEXPORT
int
mvm_madvise_free_plat(void *szone, void *r, uintptr_t pgLo, uintptr_t pgHi, uintptr_t *last, boolean_t scribble, plat_map_t *map);

MALLOC_NOEXPORT
void
mvm_protect(void * __sized_by(size) address, size_t size, unsigned protection, unsigned debug_flags);

MALLOC_NOEXPORT
void
mvm_protect_plat(void * __sized_by(size) address, size_t size, unsigned protection, unsigned debug_flags, plat_map_t *map);

#if CONFIG_MAGAZINE_DEFERRED_RECLAIM
MALLOC_NOEXPORT
kern_return_t
mvm_deferred_reclaim_init(void);

MALLOC_NOEXPORT
bool
mvm_reclaim_mark_used(mach_vm_reclaim_id_t id, mach_vm_address_t ptr, mach_vm_size_t size, unsigned int debug_flags);

MALLOC_NOEXPORT
mach_vm_reclaim_id_t
mvm_reclaim_mark_free(mach_vm_address_t ptr, mach_vm_size_t size, unsigned int debug_flags);

MALLOC_NOEXPORT
bool
mvm_reclaim_is_available(mach_vm_reclaim_id_t id);
#endif // CONFIG_MAGAZINE_DEFERRED_RECLAIM

#endif // __VM_H
