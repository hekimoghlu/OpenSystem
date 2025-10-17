/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#ifndef _VM_VM_FAULT_XNU_H_
#define _VM_VM_FAULT_XNU_H_

#ifdef XNU_KERNEL_PRIVATE

#include <sys/cdefs.h>
#include <vm/vm_fault.h>

__BEGIN_DECLS

#ifdef  MACH_KERNEL_PRIVATE

#include <vm/vm_page.h>
#include <vm/vm_object_xnu.h>
#include <vm/vm_map_xnu.h>

extern void vm_fault_init(void);

/* exported kext version */
extern kern_return_t vm_fault_external(
	vm_map_t        map,
	vm_map_offset_t vaddr,
	vm_prot_t       fault_type,
	boolean_t       change_wiring,
	int             interruptible,
	pmap_t          caller_pmap,
	vm_map_offset_t caller_pmap_addr);


extern vm_offset_t kdp_lightweight_fault(
	vm_map_t map,
	vm_offset_t cur_target_addr,
	bool multi_cpu);

#endif  /* MACH_KERNEL_PRIVATE */

/*
 * Disable vm faults on the current thread.
 */
extern void vm_fault_disable(void);

/*
 * Enable vm faults on the current thread.
 */
extern void vm_fault_enable(void);

/*
 * Return whether vm faults are disabled on the current thread.
 */
extern bool vm_fault_get_disabled(void);

extern boolean_t NEED_TO_HARD_THROTTLE_THIS_TASK(void);

__END_DECLS

#endif /* XNU_KERNEL_PRIVATE */
#endif  /* _VM_VM_FAULT_XNU_H_ */
