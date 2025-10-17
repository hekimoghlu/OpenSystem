/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#ifndef _MACH_MEMORY_ENTRY_
#define _MACH_MEMORY_ENTRY_

__BEGIN_DECLS

#if XNU_PLATFORM_MacOSX
extern kern_return_t mach_memory_entry_page_op(
	ipc_port_t              entry_port,
	vm_object_offset_ut     offset,
	int                     ops,
	ppnum_t                 *phys_entry,
	int                     *flags);

extern kern_return_t mach_memory_entry_range_op(
	ipc_port_t              entry_port,
	vm_object_offset_ut     offset_beg,
	vm_object_offset_ut     offset_end,
	int                     ops,
	int                     *range);
#endif /* XNU_PLATFORM_MacOSX */

/*
 *	Routine:	vm_convert_port_to_copy_object
 *	Purpose:
 *		Convert from a port specifying a named entry
 *              backed by a copy map to the VM object itself.
 *              Returns NULL if the port does not refer to an copy map-backed named entry.
 *	Conditions:
 *		Nothing locked.
 */
extern vm_object_t vm_convert_port_to_copy_object(
	ipc_port_t      port);

__END_DECLS

#endif /* _MACH_MEMORY_ENTRY_ */
#endif /* XNU_KERNEL_PRIVATE */
