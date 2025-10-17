/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

static inline bool
mvm_aslr_enabled(void)
{
	return _dyld_get_image_slide((const struct mach_header *)_NSGetMachExecuteHeader()) != 0;
}

MALLOC_NOEXPORT
void
mvm_aslr_init(void);

MALLOC_NOEXPORT
void *
mvm_allocate_pages(size_t size, unsigned char align, uint32_t debug_flags, int vm_page_label);

MALLOC_NOEXPORT
void
mvm_deallocate_pages(void *addr, size_t size, unsigned debug_flags);

MALLOC_NOEXPORT
int
mvm_madvise_free(void *szone, void *r, uintptr_t pgLo, uintptr_t pgHi, uintptr_t *last, boolean_t scribble);

MALLOC_NOEXPORT
void
mvm_protect(void *address, size_t size, unsigned protection, unsigned debug_flags);

#endif // __VM_H
