/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#ifndef _VM_VM_KERN_INTERNAL_H_
#define _VM_VM_KERN_INTERNAL_H_

#include <sys/cdefs.h>
#include <vm/vm_kern_xnu.h>

__BEGIN_DECLS

#ifdef MACH_KERNEL_PRIVATE

#pragma mark kmem range methods

extern struct mach_vm_range kmem_ranges[KMEM_RANGE_COUNT];
extern struct mach_vm_range kmem_large_ranges[KMEM_RANGE_COUNT];
#define KMEM_RANGE_MASK       0x3fff
#define KMEM_HASH_SET         0x4000
#define KMEM_DIRECTION_MASK   0x8000

__stateful_pure
extern mach_vm_size_t mach_vm_range_size(
	const struct mach_vm_range *r);

__attribute__((overloadable, pure))
extern bool mach_vm_range_contains(
	const struct mach_vm_range *r,
	mach_vm_offset_t        addr);

__attribute__((overloadable, pure))
extern bool mach_vm_range_contains(
	const struct mach_vm_range *r,
	mach_vm_offset_t        addr,
	mach_vm_offset_t        size);

__attribute__((overloadable, pure))
extern bool mach_vm_range_intersects(
	const struct mach_vm_range *r1,
	const struct mach_vm_range *r2);

__attribute__((overloadable, pure))
extern bool mach_vm_range_intersects(
	const struct mach_vm_range *r1,
	mach_vm_offset_t        addr,
	mach_vm_offset_t        size);

/*
 * @function kmem_range_id_contains
 *
 * @abstract Return whether the region of `[addr, addr + size)` is completely
 * within the memory range.
 */
__pure2
extern bool kmem_range_id_contains(
	kmem_range_id_t         range_id,
	vm_map_offset_t         addr,
	vm_map_size_t           size);

__pure2
extern kmem_range_id_t kmem_addr_get_range(
	vm_map_offset_t         addr,
	vm_map_size_t           size);

extern kmem_range_id_t kmem_adjust_range_id(
	uint32_t                hash);



__startup_func
extern uint16_t kmem_get_random16(
	uint16_t                upper_limit);

__startup_func
extern void kmem_shuffle(
	uint16_t               *shuffle_buf,
	uint16_t                count);

#endif /* MACH_KERNEL_PRIVATE */

__END_DECLS

#endif  /* _VM_VM_KERN_INTERNAL_H_ */
