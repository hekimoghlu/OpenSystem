/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#ifndef _IPC_IPC_EVENTLINK_H_
#define _IPC_IPC_EVENTLINK_H_

#ifdef MACH_KERNEL_PRIVATE

#include <mach/std_types.h>
#include <mach/port.h>
#include <mach/mach_eventlink_types.h>
#include <mach_assert.h>

#include <mach/mach_types.h>
#include <mach/boolean.h>
#include <mach/kern_return.h>

#include <kern/assert.h>
#include <kern/kern_types.h>

#include <ipc/ipc_types.h>
#include <ipc/ipc_object.h>
#include <ipc/ipc_port.h>
#include <kern/waitq.h>
#include <os/refcnt.h>

__options_decl(ipc_eventlink_option_t, uint64_t, {
	IPC_EVENTLINK_NONE          = 0,
	IPC_EVENTLINK_NO_WAIT       = 0x1,
	IPC_EVENTLINK_HANDOFF       = 0x2,
	IPC_EVENTLINK_FORCE_WAKEUP  = 0x4,
});

__options_decl(ipc_eventlink_type_t, uint8_t, {
	IPC_EVENTLINK_TYPE_NO_COPYIN         = 0x1,
	IPC_EVENTLINK_TYPE_WITH_COPYIN       = 0x2,
});

#define THREAD_ASSOCIATE_WILD ((struct thread *) -1)

struct ipc_eventlink_base;

struct ipc_eventlink {
	ipc_port_t                  el_port;             /* Port for eventlink object */
	thread_t                    el_thread;           /* Thread associated with eventlink object */
	struct ipc_eventlink_base   *el_base;            /* eventlink base struct */
	uint64_t                    el_sync_counter;     /* Sync counter for wait/ signal */
	uint64_t                    el_wait_counter;     /* Counter passed in eventlink wait */
};

struct ipc_eventlink_base {
	struct ipc_eventlink          elb_eventlink[2];  /* Eventlink pair */
	struct waitq                  elb_waitq;         /* waitq */
	os_refcnt_t                   elb_ref_count;     /* ref count for eventlink */
	uint8_t                       elb_type;
#if DEVELOPMENT || DEBUG
	queue_chain_t                 elb_global_elm;    /* Global list of eventlinks */
#endif
};

#define IPC_EVENTLINK_BASE_NULL ((struct ipc_eventlink_base *)NULL)
#define ipc_eventlink_active(eventlink) waitq_valid(&(eventlink)->el_base->elb_waitq)

#define eventlink_remote_side(eventlink) ((eventlink) == &((eventlink)->el_base->elb_eventlink[0]) ? \
	&((eventlink)->el_base->elb_eventlink[1]) : &((eventlink)->el_base->elb_eventlink[0]))

#define ipc_eventlink_lock(eventlink)     waitq_lock(&(eventlink)->el_base->elb_waitq)
#define ipc_eventlink_unlock(eventlink)   waitq_unlock(&(eventlink)->el_base->elb_waitq)

void ipc_eventlink_init(void);

/* Function declarations */
void
ipc_eventlink_init(void);

struct ipc_eventlink *
convert_port_to_eventlink(
	mach_port_t             port);

void
ipc_eventlink_reference(
	struct ipc_eventlink *ipc_eventlink);

void
ipc_eventlink_deallocate(
	struct ipc_eventlink *ipc_eventlink);

uint64_t
    mach_eventlink_signal_trap(
	mach_port_name_t port,
	uint64_t         signal_count __unused);

uint64_t
mach_eventlink_wait_until_trap(
	mach_port_name_t                    eventlink_port,
	uint64_t                            wait_count,
	mach_eventlink_signal_wait_option_t option,
	kern_clock_id_t                     clock_id,
	uint64_t                            deadline);

uint64_t
    mach_eventlink_signal_wait_until_trap(
	mach_port_name_t                    eventlink_port,
	uint64_t                            wait_count,
	uint64_t                            signal_count __unused,
	mach_eventlink_signal_wait_option_t option,
	kern_clock_id_t                     clock_id,
	uint64_t                            deadline);

#endif /* MACH_KERNEL_PRIVATE */
#endif /* _IPC_IPC_EVENTLINK_H_ */
