/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#ifndef _KERN_RESTARTABLE_H_
#define _KERN_RESTARTABLE_H_

#include <sys/cdefs.h>
#include <mach/message.h>
#include <mach/task.h>

__BEGIN_DECLS

/*!
 * @typedef task_restartable_range_t
 *
 * @brief
 * Describes a userspace recoverable range.
 *
 * @field location
 * The pointer to the beginning of a restartable section.
 *
 * @field length
 * The length of the critical section anchored at location.
 *
 * @field recovery_offs
 * The offset from the initial location that should be used for the recovery
 * codepath.
 *
 * @field flags
 * Currently unused, pass 0.
 */
typedef struct {
	mach_vm_address_t location;
	unsigned short    length;
	unsigned short    recovery_offs;
	unsigned int      flags;
} task_restartable_range_t;

typedef task_restartable_range_t *task_restartable_range_array_t;

/*!
 * @function task_restartable_ranges_register
 *
 * @brief
 * Register a set of restartable ranges for the current task.
 *
 * @param task
 * The task to operate on
 *
 * @param ranges
 * An array of address ranges for which PC resets are performed.
 *
 * @param count
 * The number of address ranges.
 *
 * @returns
 * - KERN_SUCCESS on success
 * - KERN_FAILURE if the task isn't the current one
 * - KERN_INVALID_ARGUMENT for various invalid inputs
 * - KERN_NOT_SUPPORTED the request is not supported (second registration on
 *   release kernels, registration when the task has gone wide)
 * - KERN_RESOURCE_SHORTAGE if not enough memory
 */
extern kern_return_t task_restartable_ranges_register(
	task_t                         task,
	task_restartable_range_array_t ranges,
	mach_msg_type_number_t         count);

/*!
 * @function task_restartable_ranges_synchronize
 *
 * @brief
 * Require for all threads in the task to reset their PC
 * if within a restartable range.
 *
 * @param task
 * The task to operate on (needs to be current task)
 *
 * @returns
 * - KERN_SUCCESS
 * - KERN_FAILURE if the task isn't the current one
 */
extern kern_return_t task_restartable_ranges_synchronize(task_t task);

/*!
 * @const TASK_RESTARTABLE_OFFSET_MAX
 * The maximum value length / recovery_offs can have.
 */
#define TASK_RESTARTABLE_OFFSET_MAX  4096u

#ifdef KERNEL_PRIVATE
#pragma GCC visibility push(hidden)

struct restartable_ranges;

/**
 * @function restartable_init
 *
 * @brief
 * Initializes the restartable module.
 */
extern void restartable_init(void);

/**
 * @function restartable_ranges_release
 *
 * @brief
 * Release a reference on a restartable range.
 */
extern void restartable_ranges_release(struct restartable_ranges *ranges);

/**
 * @function thread_reset_pcs_in_range
 *
 * @brief
 * Returns whether a non running thread is currently in a critical range.
 */
extern bool thread_reset_pcs_in_range(task_t task, struct thread *thread);

/**
 * @function thread_reset_pcs_will_fault
 *
 * @brief
 * Called by the platform code when about to handle a user fault exception.
 */
extern void thread_reset_pcs_will_fault(struct thread *thread);

/**
 * @function thread_reset_pcs_done_faulting
 *
 * @brief
 * Called by the platform code when being done handling a user fault
 * exception.
 */
extern void thread_reset_pcs_done_faulting(struct thread *thread);

/**
 * @function thread_reset_pcs_ast
 *
 * @brief
 * Perform the work at the AST boundary to reset thread PCS.
 */
extern void thread_reset_pcs_ast(task_t task, struct thread *thread);

/**
 * @function thread_reset_pcs_ack_IPI
 *
 * @brief
 * Called by the scheduler code when acking the reset-pcs IPI.
 */
extern void thread_reset_pcs_ack_IPI(struct thread *thread);

#pragma GCC visibility pop
#endif // KERNEL_PRIVATE

__END_DECLS

#endif  /* _KERN_RESTARTABLE_H_ */
