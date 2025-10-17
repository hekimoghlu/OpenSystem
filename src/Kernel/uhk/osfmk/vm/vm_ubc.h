/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#ifndef _VM_VM_UBC_H_
#define _VM_VM_UBC_H_

#include <sys/cdefs.h>
#include <mach/memory_object_types.h>
#include <mach/mach_types.h>
#include <kern/kern_types.h>
#include <vm/vm_options.h>

/*
 * The upl declarations are all usable by ubc
 */
#include <vm/vm_upl.h>
__BEGIN_DECLS

struct vnode;

extern struct vnode * upl_lookup_vnode(upl_t upl);

extern upl_t vector_upl_create(vm_offset_t, uint32_t);
extern upl_size_t vector_upl_get_size(const upl_t);
extern boolean_t vector_upl_is_valid(upl_t);
extern boolean_t vector_upl_set_subupl(upl_t, upl_t, u_int32_t);
extern void vector_upl_set_pagelist(upl_t);
uint32_t vector_upl_max_upls(const upl_t upl);


extern kern_return_t    memory_object_pages_resident(
	memory_object_control_t         control,
	boolean_t                       *               has_pages_resident);

extern kern_return_t    memory_object_signed(
	memory_object_control_t         control,
	boolean_t                       is_signed);

extern boolean_t        memory_object_is_signed(
	memory_object_control_t control);

extern void             memory_object_mark_used(
	memory_object_control_t         control);

extern void             memory_object_mark_unused(
	memory_object_control_t         control,
	boolean_t                       rage);

extern void             memory_object_mark_io_tracking(
	memory_object_control_t         control);

extern void             memory_object_mark_trusted(
	memory_object_control_t         control);


extern memory_object_t vnode_pager_setup(
	struct vnode *, memory_object_t);

extern void vnode_pager_deallocate(
	memory_object_t);
extern void vnode_pager_vrele(
	struct vnode *vp);

extern kern_return_t memory_object_create_named(
	memory_object_t pager,
	memory_object_offset_t  size,
	memory_object_control_t         *control);

typedef int pager_return_t;
extern pager_return_t   vnode_pagein(
	struct vnode *, upl_t,
	upl_offset_t, vm_object_offset_t,
	upl_size_t, int, int *);
extern pager_return_t   vnode_pageout(
	struct vnode *, upl_t,
	upl_offset_t, vm_object_offset_t,
	upl_size_t, int, int *);

__END_DECLS

#endif /* _VM_VM_UBC_H_ */
