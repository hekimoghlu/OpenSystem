/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
 *
 */
/*
 *	File:	kern/sync_sema.h
 *	Author:	Joseph CaraDonna
 *
 *	Contains RT distributed semaphore synchronization service definitions.
 */

#ifndef _KERN_SYNC_SEMA_H_
#define _KERN_SYNC_SEMA_H_

#include <kern/kern_types.h>
#include <mach/sync_policy.h>
#include <mach/clock_types.h>

#ifdef MACH_KERNEL_PRIVATE

#include <kern/queue.h>
#include <kern/waitq.h>
#include <os/refcnt.h>

typedef struct semaphore {
	queue_chain_t     task_link;  /* chain of semaphores owned by a task */
	struct waitq      waitq;      /* queue of blocked threads & lock     */
	task_t            owner;      /* task that owns semaphore            */
	ipc_port_t        port;       /* semaphore port                      */
	os_ref_atomic_t   ref_count;  /* reference count                     */
	int               count;      /* current count value                 */
} Semaphore;

#define semaphore_lock(semaphore)   waitq_lock(&(semaphore)->waitq)
#define semaphore_unlock(semaphore) waitq_unlock(&(semaphore)->waitq)

extern void semaphore_reference(
	semaphore_t semaphore);

extern void semaphore_dereference(
	semaphore_t semaphore);

#pragma GCC visibility push(hidden)

extern void semaphore_destroy_all(
	task_t      task);

extern semaphore_t convert_port_to_semaphore(
	ipc_port_t  port);

extern ipc_port_t convert_semaphore_to_port(
	semaphore_t semaphore);

extern kern_return_t port_name_to_semaphore(
	mach_port_name_t  name,
	semaphore_t       *semaphore);

#pragma GCC visibility pop
#endif /* MACH_KERNEL_PRIVATE */
#if XNU_KERNEL_PRIVATE
#pragma GCC visibility push(hidden)

#define SEMAPHORE_CONT_NULL ((semaphore_cont_t)NULL)
typedef void (*semaphore_cont_t)(kern_return_t);

extern kern_return_t semaphore_signal_internal_trap(
	mach_port_name_t sema_name);

extern kern_return_t semaphore_timedwait_signal_trap_internal(
	mach_port_name_t wait_name,
	mach_port_name_t signal_name,
	unsigned int     sec,
	clock_res_t      nsec,
	semaphore_cont_t);

extern kern_return_t semaphore_timedwait_trap_internal(
	mach_port_name_t name,
	unsigned int     sec,
	clock_res_t      nsec,
	semaphore_cont_t);

extern kern_return_t semaphore_wait_signal_trap_internal(
	mach_port_name_t wait_name,
	mach_port_name_t signal_name,
	semaphore_cont_t);

extern kern_return_t semaphore_wait_trap_internal(
	mach_port_name_t name,
	semaphore_cont_t);

#pragma GCC visibility pop
#endif /* XNU_KERNEL_PRIVATE */
#endif /* _KERN_SYNC_SEMA_H_ */
