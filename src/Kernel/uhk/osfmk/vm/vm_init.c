/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
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
 *	File:	vm/vm_init.c
 *	Author:	Avadis Tevanian, Jr., Michael Wayne Young
 *	Date:	1985
 *
 *	Initialize the Virtual Memory subsystem.
 */

#include <mach/machine/vm_types.h>
#include <mach/vm_map.h>
#include <kern/startup.h>
#include <kern/zalloc_internal.h>
#include <kern/kext_alloc.h>
#include <sys/kdebug.h>
#include <vm/vm_object_internal.h>
#include <vm/vm_map_internal.h>
#include <vm/vm_page_internal.h>
#include <vm/vm_kern.h>
#include <vm/memory_object.h>
#include <vm/vm_fault_xnu.h>
#include <vm/vm_init_xnu.h>

#include <pexpert/pexpert.h>

#include <vm/vm_protos.h>

const vm_offset_t vm_min_kernel_address = VM_MIN_KERNEL_AND_KEXT_ADDRESS;
const vm_offset_t vm_max_kernel_address = VM_MAX_KERNEL_ADDRESS;

TUNABLE(bool, iokit_iomd_setownership_enabled,
    "iokit_iomd_setownership_enabled", true);

static inline void
vm_mem_bootstrap_log(const char *message)
{
//	kprintf("vm_mem_bootstrap: %s\n", message);
	kernel_debug_string_early(message);
}

/*
 *	vm_mem_bootstrap initializes the virtual memory system.
 *	This is done only by the first cpu up.
 */
__startup_func
void
vm_mem_bootstrap(void)
{
	vm_offset_t start, end;

	/*
	 *	Initializes resident memory structures.
	 *	From here on, all physical memory is accounted for,
	 *	and we use only virtual addresses.
	 */
	vm_mem_bootstrap_log("vm_page_bootstrap");
	vm_page_bootstrap(&start, &end);

	/*
	 *	Initialize other VM packages
	 */

	vm_mem_bootstrap_log("zone_bootstrap");
	zone_bootstrap();

	vm_mem_bootstrap_log("vm_object_bootstrap");
	vm_object_bootstrap();

	vm_retire_boot_pages();


	vm_mem_bootstrap_log("vm_map_init");
	vm_map_init();

	vm_mem_bootstrap_log("kmem_init");
	kmem_init(start, end);

	kernel_startup_initialize_upto(STARTUP_SUB_KMEM);

	vm_mem_bootstrap_log("vm_fault_init");
	vm_fault_init();

	kernel_startup_initialize_upto(STARTUP_SUB_ZALLOC);

	if (iokit_iomd_setownership_enabled) {
		kprintf("IOKit IOMD setownership ENABLED\n");
	} else {
		kprintf("IOKit IOMD setownership DISABLED\n");
	}
}
