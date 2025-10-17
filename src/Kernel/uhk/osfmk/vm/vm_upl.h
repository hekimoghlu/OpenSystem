/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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

#ifndef _VM_UPL_
#define _VM_UPL_

#include <mach/vm_types.h>
#include <mach/vm_prot.h>
#include <mach/kern_return.h>
#include <mach/memory_object_types.h>

__BEGIN_DECLS
/*
 * VM routines that used to be published to
 * user space, and are now restricted to the kernel.
 *
 * They should eventually go away entirely -
 * to be replaced with standard vm_map() and
 * vm_deallocate() calls.
 */
extern kern_return_t vm_upl_map
(
	vm_map_t target_task,
	upl_t upl,
	vm_address_t *address
);

extern kern_return_t vm_upl_unmap
(
	vm_map_t target_task,
	upl_t upl
);

extern kern_return_t vm_upl_map_range
(
	vm_map_t target_task,
	upl_t upl,
	vm_offset_t offset,
	vm_size_t size,
	vm_prot_t prot,
	vm_address_t *address
);

extern kern_return_t vm_upl_unmap_range
(
	vm_map_t target_task,
	upl_t upl,
	vm_offset_t offset,
	vm_size_t size
);

/* Support for UPLs from vm_maps */
extern kern_return_t vm_map_get_upl(
	vm_map_t                target_map,
	vm_map_offset_t         map_offset,
	upl_size_t              *size,
	upl_t                   *upl,
	upl_page_info_array_t   page_info,
	unsigned int            *page_infoCnt,
	upl_control_flags_t     *flags,
	vm_tag_t                tag,
	int                     force_data_sync);

__END_DECLS

#endif /* _VM_UPL_ */
#endif /* XNU_KERNEL_PRIVATE */
