/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#ifdef  XNU_KERNEL_PRIVATE

#ifndef _VM_VM_COMPRESSOR_PAGER_XNU_H_
#define _VM_VM_COMPRESSOR_PAGER_XNU_H_

#include <mach/mach_types.h>
#include <kern/kern_types.h>
#include <vm/vm_external.h>

__options_decl(vm_compressor_options_t, uint32_t, {
	C_DONT_BLOCK            = 0x00000001, /* vm_fault tells the compressor not to read from swap file */
	C_KEEP                  = 0x00000002, /* vm_fault tells the compressor to not remove the data from the segment after decompress*/
	C_KDP                   = 0x00000004, /* kdp fault tells the compressor to not do locking */
	C_PAGE_UNMODIFIED       = 0x00000008,
	C_KDP_MULTICPU          = 0x00000010,
});

extern kern_return_t vm_compressor_pager_get(
	memory_object_t         mem_obj,
	memory_object_offset_t  offset,
	ppnum_t                 ppnum,
	int                     *my_fault_type,
	vm_compressor_options_t flags,
	int                     *compressed_count_delta_p);


#if CONFIG_TRACK_UNMODIFIED_ANON_PAGES
extern uint64_t compressor_ro_uncompressed;
extern uint64_t compressor_ro_uncompressed_total_returned;
extern uint64_t compressor_ro_uncompressed_skip_returned;
extern uint64_t compressor_ro_uncompressed_get;
extern uint64_t compressor_ro_uncompressed_put;
extern uint64_t compressor_ro_uncompressed_swap_usage;
#endif /* CONFIG_TRACK_UNMODIFIED_ANON_PAGES */

extern unsigned int vm_compressor_pager_get_count(memory_object_t mem_obj);

#endif  /* _VM_VM_COMPRESSOR_PAGER_XNU_H_ */

#endif  /* XNU_KERNEL_PRIVATE */
