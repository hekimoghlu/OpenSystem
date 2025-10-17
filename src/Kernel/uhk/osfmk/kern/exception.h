/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#ifndef _KERN_EXCEPTION_H_
#define _KERN_EXCEPTION_H_

#include <mach/mach_types.h>
#include <mach/thread_status.h>
#include <mach/exception_types.h>
#include <kern/kern_types.h>
#include <security/_label.h>

/*
 * Common storage for exception actions.
 * There are arrays of these maintained at the activation, task, and host.
 */
struct exception_action {
	struct ipc_port         * XNU_PTRAUTH_SIGNED_PTR("exception_action.port") port; /* exception port */
	thread_state_flavor_t   flavor;         /* state flavor to send */
	exception_behavior_t    behavior;       /* exception type to raise */
	boolean_t               privileged;     /* survives ipc_task_reset */
	boolean_t               hardened;       /* associated with the task's hardened_exception_action */
	struct label            *label;         /* MAC label associated with action */
};
struct hardened_exception_action {
	struct exception_action ea;
	uint32_t                signed_pc_key;
	exception_mask_t        exception;
};

/* Initialize global state needed for exceptions. */
extern void exception_init(void);

extern ipc_port_t exception_port_copy_send(ipc_port_t port);

/* Make an up-call to a thread's exception server */
extern kern_return_t exception_triage(
	exception_type_t        exception,
	mach_exception_data_t   code,
	mach_msg_type_number_t  codeCnt);

extern kern_return_t exception_triage_thread(
	exception_type_t        exception,
	mach_exception_data_t   code,
	mach_msg_type_number_t  codeCnt,
	thread_t                thread);

#define BT_EXC_PORTS_COUNT 3
extern void exception_deliver_backtrace(
	kcdata_object_t bt_object,
	ipc_port_t      exc_ports[static BT_EXC_PORTS_COUNT],
	exception_type_t exception);

/* Notify system performance monitor */
extern kern_return_t sys_perf_notify(thread_t thread, int pid);

/* Notify crash reporter */
extern kern_return_t task_exception_notify(exception_type_t exception,
    mach_exception_data_type_t code, mach_exception_data_type_t subcode, const bool fatal);

#endif  /* _KERN_EXCEPTION_H_ */
