/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#ifndef __VM_RECLAIM_XNU__
#define __VM_RECLAIM_XNU__

#if XNU_KERNEL_PRIVATE
#if CONFIG_DEFERRED_RECLAIM
#include <mach/mach_types.h>
#if BSD_KERNEL_PRIVATE
#include <mach/mach_vm.h>
#else /* BSD_KERNEL_PRIVATE */
#include <mach/mach_vm_server.h>
#endif /* BSD_KERNEL_PRIVATE */

typedef struct vm_deferred_reclamation_metadata_s *vm_deferred_reclamation_metadata_t;

__enum_closed_decl(vm_deferred_reclamation_gc_action_t, uint8_t, {
	/* Trim all buffers */
	RECLAIM_GC_TRIM = 0,
	/* Fully drain all buffers */
	RECLAIM_GC_DRAIN = 1,
	/* Drain any buffers belonging to suspended tasks */
	RECLAIM_GC_SCAVENGE = 2,
});

__options_closed_decl(vm_deferred_reclamation_options_t, uint8_t, {
	RECLAIM_OPTIONS_NONE = 0x00,
	/* Do not fault on the reclaim buffer if it is not resident */
	RECLAIM_NO_FAULT     = 0x01,
	/* Do not wait to acquire the buffer if it is owned by another thread */
	RECLAIM_NO_WAIT      = 0x02,
});

/*
 * Deallocate the kernel metadata associated with this reclamation buffer
 * Note that this does NOT free the memory in the buffer.
 * This is called from the task_destroy path, so we're about to reclaim all of the task's memory
 * anyways.
 */
void vm_deferred_reclamation_buffer_deallocate(vm_deferred_reclamation_metadata_t metadata);

/*
 * Synchronously drain all reclamation ring's belonging to a task.
 */
kern_return_t vm_deferred_reclamation_task_drain(
	task_t                            task,
	vm_deferred_reclamation_options_t options);

/*
 * Return true if this task has a reclamation ring
 */
bool vm_deferred_reclamation_task_has_ring(task_t task);

/*
 * Create a fork of the given reclamation buffer for a new task.
 * Parent buffer must be locked and will be unlocked on return.
 *
 * This must be called when forking a task that has a reclamation buffer
 * to duplicate the parent's buffer, and the new buffer must later be
 * registered by vm_deferred_reclamation_task_fork_register.
 *
 * The caller must lock the parent's reclamation buffer BEFORE forking
 * the parent's vm_map. Otherwise the parent's buffer could get reclaimed
 * in between the map fork and the buffer fork causing the child's
 * data strucutres to be out of sync.
 */
vm_deferred_reclamation_metadata_t vm_deferred_reclamation_task_fork(
	task_t task,
	vm_deferred_reclamation_metadata_t parent);

/*
 * Add a reclamation buffer returned by vm_deferred_reclamation_task_fork to
 * the global reclamation queues.
 *
 * The child task should call this to ensure that the kernel knows about its
 * reclamation buffer. This must happen after the child's address space is
 * fully initialized and able to recieve VM API calls.
 */
void
vm_deferred_reclamation_task_fork_register(vm_deferred_reclamation_metadata_t metadata);

/*
 * Set the current thread as the owner of a reclaim buffer. May block. Will
 * propagate priority. Should be called before forking the owning task.
 */
void vm_deferred_reclamation_ring_own(vm_deferred_reclamation_metadata_t metadata);

/*
 * Release ownership of a reclaim buffer and wakeup any threads waiting for
 * ownership. Must be called from the thread that acquired ownership.
 */
void vm_deferred_reclamation_ring_disown(vm_deferred_reclamation_metadata_t metadata);

/*
 * Should be called when a task is suspended -- will trigger asynchronous
 * reclamation of any reclaim rings owned by the task.
 */
void vm_deferred_reclamation_task_suspend(task_t task);

/*
 * Perform Garbage Collection on all reclaim rings
 */
void vm_deferred_reclamation_gc(vm_deferred_reclamation_gc_action_t action,
    vm_deferred_reclamation_options_t options);

#endif /* CONFIG_DEFERRED_RECLAIM */
#endif /* XNU_KERNEL_PRIVATE */
#endif  /* __VM_RECLAIM_XNU__ */
