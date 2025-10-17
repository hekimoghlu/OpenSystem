/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
#ifndef _VM_VM_FAULT_INTERNAL_H_
#define _VM_VM_FAULT_INTERNAL_H_


#include <sys/cdefs.h>
#include <vm/vm_fault_xnu.h>

__BEGIN_DECLS

#ifdef  MACH_KERNEL_PRIVATE

/*
 *	Page fault handling based on vm_object only.
 */

extern vm_fault_return_t vm_fault_page(
	/* Arguments: */
	vm_object_t     first_object,           /* Object to begin search */
	vm_object_offset_t first_offset,        /* Offset into object */
	vm_prot_t       fault_type,             /* What access is requested */
	boolean_t       must_be_resident,        /* Must page be resident? */
	boolean_t       caller_lookup,          /* caller looked up page */
	/* Modifies in place: */
	vm_prot_t       *protection,            /* Protection for mapping */
	vm_page_t       *result_page,           /* Page found, if successful */
	/* Returns: */
	vm_page_t       *top_page,              /* Page in top object, if
                                                 * not result_page.  */
	int             *type_of_fault,         /* if non-zero, return COW, zero-filled, etc...
                                                 * used by kernel trace point in vm_fault */
	/* More arguments: */
	kern_return_t   *error_code,            /* code if page is in error */
	boolean_t       no_zero_fill,           /* don't fill absent pages */
	vm_object_fault_info_t fault_info);

extern void vm_fault_cleanup(
	vm_object_t     object,
	vm_page_t       top_page);

extern kern_return_t vm_fault_wire(
	vm_map_t        map,
	vm_map_entry_t  entry,
	vm_prot_t       prot,
	vm_tag_t        wire_tag,
	pmap_t          pmap,
	vm_map_offset_t pmap_addr,
	ppnum_t         *physpage_p);

extern void vm_fault_unwire(
	vm_map_t        map,
	vm_map_entry_t  entry,
	boolean_t       deallocate,
	pmap_t          pmap,
	vm_map_offset_t pmap_addr,
	vm_map_offset_t end_addr);

extern kern_return_t    vm_fault_copy(
	vm_object_t             src_object,
	vm_object_offset_t      src_offset,
	vm_map_size_t           *copy_size,             /* INOUT */
	vm_object_t             dst_object,
	vm_object_offset_t      dst_offset,
	vm_map_t                dst_map,
	vm_map_version_t         *dst_version,
	int                     interruptible);

extern kern_return_t vm_fault_enter(
	vm_page_t m,
	pmap_t pmap,
	vm_map_offset_t vaddr,
	vm_map_size_t fault_page_size,
	vm_map_offset_t fault_phys_offset,
	vm_prot_t prot,
	vm_prot_t fault_type,
	boolean_t wired,
	vm_tag_t  wire_tag,             /* if wiring must pass tag != VM_KERN_MEMORY_NONE */
	vm_object_fault_info_t fault_info,
	boolean_t *need_retry,
	int *type_of_fault,
	uint8_t *object_lock_type);

extern kern_return_t vm_pre_fault_with_info(
	vm_map_t                map,
	vm_map_offset_t         offset,
	vm_prot_t               prot,
	vm_object_fault_info_t  fault_info);

#endif /* MACH_KERNEL_PRIVATE */

__END_DECLS

#endif  /* _VM_VM_FAULT_INTERNAL_H_ */
