/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
 * Copyright (c) 1991,1990 Carnegie Mellon University
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

#include <mach/vm_param.h>
#include <vm/vm_kern_xnu.h>
#include <vm/vm_map_xnu.h>
#include <i386/pmap.h>
#include <san/kasan.h>

extern vm_offset_t virtual_avail;

#define IO_MAP_SIZE            (32ul << 20)

__startup_data static struct mach_vm_range io_range;
static SECURITY_READ_ONLY_LATE(vm_map_t) io_submap;
KMEM_RANGE_REGISTER_STATIC(io_submap, &io_range, IO_MAP_SIZE);

__startup_func
static void
io_map_init(void)
{
	vm_map_will_allocate_early_map(&io_submap);
	io_submap = kmem_suballoc(kernel_map, &io_range.min_address, IO_MAP_SIZE,
	    VM_MAP_CREATE_NEVER_FAULTS | VM_MAP_CREATE_DISABLE_HOLELIST,
	    VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, KMS_PERMANENT | KMS_NOFAIL,
	    VM_KERN_MEMORY_IOKIT).kmr_submap;
}
STARTUP(KMEM, STARTUP_RANK_LAST, io_map_init);

/*
 * Allocate and map memory for devices that may need to be mapped before
 * Mach VM is running.
 */
vm_offset_t
io_map(
	vm_map_offset_t         phys_addr,
	vm_size_t               size,
	unsigned int            flags,
	vm_prot_t               prot,
	bool                    unmappable)
{
	vm_offset_t start_offset = phys_addr - trunc_page(phys_addr);
	vm_offset_t alloc_size   = round_page(size + start_offset);
	vm_offset_t start;

	phys_addr = trunc_page(phys_addr);

	if (startup_phase < STARTUP_SUB_KMEM) {
		/*
		 * VM is not initialized.  Grab memory.
		 */
		start = virtual_avail;
		virtual_avail += round_page(size);

		pmap_map_bd(start, phys_addr, phys_addr + alloc_size, prot, flags);
#if KASAN
		kasan_notify_address(start, size);
#endif
	} else {
		kma_flags_t kmaflags = KMA_NOFAIL | KMA_PAGEABLE;

		if (unmappable) {
			kmaflags |= KMA_DATA;
		} else {
			kmaflags |= KMA_PERMANENT;
		}

		kmem_alloc(unmappable ? kernel_map : io_submap,
		    &start, alloc_size, kmaflags, VM_KERN_MEMORY_IOKIT);
		pmap_map(start, phys_addr, phys_addr + alloc_size, prot, flags);
	}
	return start + start_offset;
}
