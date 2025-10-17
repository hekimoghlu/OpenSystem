/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#ifndef _VM_VM_COMPRESSOR_PAGER_INTERNAL_H_
#define _VM_VM_COMPRESSOR_PAGER_INTERNAL_H_

#include <sys/cdefs.h>
#include <vm/vm_compressor_pager_xnu.h>

__BEGIN_DECLS
#ifdef XNU_KERNEL_PRIVATE

extern kern_return_t vm_compressor_pager_put(
	memory_object_t                 mem_obj,
	memory_object_offset_t          offset,
	ppnum_t                         ppnum,
	void                            **current_chead,
	char                            *scratch_buf,
	int                             *compressed_count_delta_p,
	vm_compressor_options_t         flags);


extern unsigned int vm_compressor_pager_state_clr(
	memory_object_t         mem_obj,
	memory_object_offset_t  offset);
extern vm_external_state_t vm_compressor_pager_state_get(
	memory_object_t         mem_obj,
	memory_object_offset_t  offset);

extern void vm_compressor_pager_transfer(
	memory_object_t         dst_mem_obj,
	memory_object_offset_t  dst_offset,
	memory_object_t         src_mem_obj,
	memory_object_offset_t  src_offset);
extern memory_object_offset_t vm_compressor_pager_next_compressed(
	memory_object_t         mem_obj,
	memory_object_offset_t  offset);

__enum_closed_decl(vm_decompress_result_t, int, {
	DECOMPRESS_SUCCESS_SWAPPEDIN = 1,
	DECOMPRESS_SUCCESS = 0,
	DECOMPRESS_NEED_BLOCK = -2,
	DECOMPRESS_FIRST_FAIL_CODE = -3,
	DECOMPRESS_FAILED_BAD_Q = -3,
	DECOMPRESS_FAILED_BAD_Q_FREEZE = -4,
	DECOMPRESS_FAILED_ALGO_ERROR = -5,
	DECOMPRESS_FAILED_WKDMD_POPCNT = -6,
	DECOMPRESS_FAILED_UNMODIFIED = -7,
});

extern bool osenvironment_is_diagnostics(void);
extern void vm_compressor_init(void);
extern bool vm_compressor_is_slot_compressed(int *slot);
extern kern_return_t vm_compressor_put(ppnum_t pn, int *slot, void **current_chead, char *scratch_buf, vm_compressor_options_t flags);
extern vm_decompress_result_t vm_compressor_get(ppnum_t pn, int *slot, vm_compressor_options_t flags);
extern int vm_compressor_free(int *slot, vm_compressor_options_t flags);

#if CONFIG_TRACK_UNMODIFIED_ANON_PAGES
extern int vm_uncompressed_put(ppnum_t pn, int *slot);
extern int vm_uncompressed_get(ppnum_t pn, int *slot, vm_compressor_options_t flags);
extern int vm_uncompressed_free(int *slot, vm_compressor_options_t flags);
#endif /* CONFIG_TRACK_UNMODIFIED_ANON_PAGES */
extern unsigned int vm_compressor_pager_reap_pages(memory_object_t mem_obj, vm_compressor_options_t flags);

extern void vm_compressor_pager_count(memory_object_t mem_obj,
    int compressed_count_delta,
    boolean_t shared_lock,
    vm_object_t object);

extern void vm_compressor_transfer(int *dst_slot_p, int *src_slot_p);

#if CONFIG_FREEZE
extern kern_return_t vm_compressor_pager_relocate(memory_object_t mem_obj, memory_object_offset_t mem_offset, void **current_chead);
extern kern_return_t vm_compressor_relocate(void **current_chead, int *src_slot_p);
extern void vm_compressor_finished_filling(void **current_chead);
#endif /* CONFIG_FREEZE */

#if DEVELOPMENT || DEBUG
extern kern_return_t vm_compressor_pager_inject_error(memory_object_t pager,
    memory_object_offset_t offset);
extern void vm_compressor_inject_error(int *slot);

extern kern_return_t vm_compressor_pager_dump(memory_object_t mem_obj, char *buf, size_t *size,
    bool *is_compressor, unsigned int *slot_count);
#endif /* DEVELOPMENT || DEBUG */

#endif /* XNU_KERNEL_PRIVATE */
__END_DECLS

#endif  /* _VM_VM_COMPRESSOR_PAGER_INTERNAL_H_ */
