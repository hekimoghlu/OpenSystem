/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
/*
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */
/*
 *	File:	memory_object.h
 *	Author:	Michael Wayne Young
 *
 *	External memory management interface definition.
 */

#ifndef _MACH_MEMORY_OBJECT_H_
#define _MACH_MEMORY_OBJECT_H_

#include <mach/vm_types.h>
#include <mach/memory_object_types.h>

__BEGIN_DECLS
#pragma GCC visibility push(hidden)

/*
 *	Initialize the specified memory object, providing
 *	a memory object control reference on which to make
 *	cache control calls.
 *	[To allow the mapping of this object to be used, the
 *	memory manager must call memory_object_set_attributes,
 *	specifying the "ready" parameter as TRUE.  To reject
 *	all mappings of this object, the memory manager may
 *	use memory_object_destroy.]
 */
extern kern_return_t memory_object_init(
	memory_object_t memory_object,
	memory_object_control_t memory_control,
	memory_object_cluster_size_t memory_object_page_size);

/*
 *	Indicates that the specified memory object is no longer
 *	mapped (or cached -- see memory_object_set_attributes),
 *	and that further mappings will cause another memory_object_init
 *	call to be made.
 *
 *	[The kernel will release its reference on the memory object
 *	after this call returns.  The memory object control associated
 *	with the memory object is no longer usable - the pager should
 *	drop the control reference granted to it by memory_object_init.]
 */
extern kern_return_t memory_object_terminate(
	memory_object_t memory_object);

/*
 *	Request data from this memory object.  At least
 *	the specified data should be returned with at
 *	least the specified access permitted.
 *
 *	[Response should be upl commit over the specified range.]
 */
extern kern_return_t memory_object_data_request(
	memory_object_t memory_object,
	memory_object_offset_t offset,
	memory_object_cluster_size_t length,
	vm_prot_t desired_access,
	memory_object_fault_info_t fault_info);

/*
 *	Return data to manager.  This call is used in place of data_write
 *	for objects initialized by object_ready instead of set_attributes.
 *	This call indicates whether the returned data is dirty and whether
 *	the kernel kept a copy.  Precious data remains precious if the
 *	kernel keeps a copy.  The indication that the kernel kept a copy
 *	is only a hint if the data is not precious; the cleaned copy may
 *	be discarded without further notifying the manager.
 *
 *	[response should be a upl_commit over the range specified]
 */
extern kern_return_t memory_object_data_return(
	memory_object_t memory_object,
	memory_object_offset_t offset,
	memory_object_cluster_size_t size,
	memory_object_offset_t *resid_offset,
	int *io_error,
	boolean_t dirty,
	boolean_t kernel_copy,
	int upl_flags);

/*
 *	Provide initial data contents for this region of
 *	the memory object.  If data has already been written
 *	to the object, this value must be discarded; otherwise,
 *	this call acts identically to memory_object_data_return.
 *
 *	[response should be UPL commit over the specified range.]
 */
extern kern_return_t memory_object_data_initialize(
	memory_object_t memory_object,
	memory_object_offset_t offset,
	memory_object_cluster_size_t size);

/*
 *	Notify the pager that the specified memory object
 *	has no other (mapped) references besides the named
 *	reference held by the pager itself.
 *
 *	[Response should be a release of the named reference when
 *	the pager deems that appropriate.]
 */
extern kern_return_t memory_object_map(
	memory_object_t memory_object,
	vm_prot_t prot);

extern kern_return_t memory_object_last_unmap(
	memory_object_t memory_object);

#pragma GCC visibility pop
__END_DECLS

#endif  /* _MACH_MEMORY_OBJECT_H_ */
