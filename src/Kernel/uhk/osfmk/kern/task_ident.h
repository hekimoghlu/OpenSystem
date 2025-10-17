/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
 *
 * A task identity token represents the identity of a mach task without carrying task
 * access capabilities. In applicable scenarios, task identity token can be moved between
 * tasks and be upgraded to desired level of task port flavor (namely, task name port,
 * inspect port, read port or control port) upon use.
 *
 */

#ifndef _KERN_TASK_IDENT_H
#define _KERN_TASK_IDENT_H

#if KERNEL_PRIVATE

#include <kern/kern_types.h>
#include <mach/mach_types.h>

__BEGIN_DECLS

#define TASK_IDENTITY_TOKEN_KPI_VERSION 1

/*!
 * @function task_id_token_port_name_to_task()
 *
 * @abstract
 * Produces a task reference from task identity token port name.
 *
 * For export to kexts only. _DO NOT_ use in kenel proper for correct task
 * reference counting.
 *
 * @param name     port name for the task identity token to operate on
 * @param taskp    task_t pointer
 *
 * @returns        KERN_SUCCESS           A valid task reference is produced.
 *                 KERN_NOT_FOUND         Cannot find task represented by token.
 *                 KERN_INVALID_ARGUMENT  Passed identity token is invalid.
 */
#if XNU_KERNEL_PRIVATE
kern_return_t task_id_token_port_name_to_task(mach_port_name_t name, task_t *taskp)
__XNU_INTERNAL(task_id_token_port_name_to_task);

struct proc_ident {
	uint64_t        p_uniqueid;
	pid_t           p_pid;
	int             p_idversion;
};

struct task_id_token {
	struct proc_ident ident;
	ipc_port_t        port;
	uint64_t          task_uniqueid; /* for corpse task */
	os_refcnt_t       tidt_refs;
};

void task_id_token_release(task_id_token_t token);

ipc_port_t convert_task_id_token_to_port(task_id_token_t token);

task_id_token_t convert_port_to_task_id_token(ipc_port_t port);

#if MACH_KERNEL_PRIVATE
kern_return_t task_identity_token_get_task_grp(task_id_token_t token, task_t *taskp, task_grp_t grp);
#endif /* MACH_KERNEL_PRIVATE */

#else  /* !XNU_KERNEL_PRIVATE */
kern_return_t task_id_token_port_name_to_task(mach_port_name_t name, task_t *taskp);
#endif /* !XNU_KERNEL_PRIVATE */

__END_DECLS

#endif /* KERNEL_PRIVATE */

#endif /* _KERN_TASK_IDENT_H */
