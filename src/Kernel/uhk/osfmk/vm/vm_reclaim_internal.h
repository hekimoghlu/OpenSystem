/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#ifndef __VM_RECLAIM_INTERNAL__
#define __VM_RECLAIM_INTERNAL__

#if CONFIG_DEFERRED_RECLAIM

#include <mach/mach_types.h>
#include <mach/vm_reclaim.h>
#include <sys/cdefs.h>
#include <vm/vm_reclaim_xnu.h>

#if MACH_KERNEL_PRIVATE

mach_error_t vm_deferred_reclamation_buffer_allocate_internal(
	task_t            task,
	mach_vm_address_ut *address,
	mach_vm_reclaim_count_t len,
	mach_vm_reclaim_count_t max_len);

kern_return_t vm_deferred_reclamation_buffer_flush_internal(
	task_t                  task,
	mach_vm_reclaim_count_t max_entries_to_reclaim);

kern_return_t vm_deferred_reclamation_buffer_update_reclaimable_bytes_internal(
	task_t task, uint64_t reclaimable_bytes);

/*
 * Resize the reclaim buffer for a given task
 */
kern_return_t vm_deferred_reclamation_buffer_resize_internal(
	task_t            task,
	mach_vm_reclaim_count_t len);


void vm_deferred_reclamation_buffer_lock(vm_deferred_reclamation_metadata_t metadata);
void vm_deferred_reclamation_buffer_unlock(vm_deferred_reclamation_metadata_t metadata);

#if DEVELOPMENT || DEBUG
/*
 * Testing helpers
 */
bool vm_deferred_reclamation_block_until_task_has_been_reclaimed(task_t task);
#endif /* DEVELOPMENT || DEBUG */

#endif /* MACH_KERNEL_PRIVATE */
#endif /* CONFIG_DEFERRED_RECLAIM */
#endif /*__VM_RECLAIM_INTERNAL__ */
