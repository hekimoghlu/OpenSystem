/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
 *	File:		kern/ipc_clock.c
 *	Purpose:	Routines to support ipc semantics of new kernel
 *			alarm clock facility.
 */

#include <mach/message.h>
#include <kern/host.h>
#include <kern/processor.h>
#include <kern/task.h>
#include <kern/thread.h>
#include <kern/ipc_host.h>
#include <kern/ipc_kobject.h>
#include <kern/clock.h>
#include <kern/misc_protos.h>
#include <ipc/ipc_port.h>
#include <ipc/ipc_space.h>

IPC_KOBJECT_DEFINE(IKOT_CLOCK,
    .iko_op_stable    = true,
    .iko_op_permanent = true);

/*
 *	Routine:	ipc_clock_init
 *	Purpose:
 *		Initialize ipc control of a clock.
 */
void
ipc_clock_init(clock_t clock)
{
	clock->cl_service = ipc_kobject_alloc_port(clock, IKOT_CLOCK,
	    IPC_KOBJECT_ALLOC_NONE);
}

/*
 *	Routine:	convert_port_to_clock
 *	Purpose:
 *		Convert from a port to a clock.
 *		Doesn't consume the port ref
 *		which may be null.
 *	Conditions:
 *		Nothing locked.
 */
clock_t
convert_port_to_clock(ipc_port_t port)
{
	clock_t clock = CLOCK_NULL;

	if (IP_VALID(port)) {
		clock = ipc_kobject_get_stable(port, IKOT_CLOCK);
	}

	return clock;
}

/*
 *	Routine:	convert_clock_to_port
 *	Purpose:
 *		Convert from a clock to a port.
 *		Produces a naked send right which may be invalid.
 *	Conditions:
 *		Nothing locked.
 */
ipc_port_t
convert_clock_to_port(clock_t clock)
{
	return ipc_kobject_make_send(clock->cl_service, clock, IKOT_CLOCK);
}

/*
 *	Routine:	port_name_to_clock
 *	Purpose:
 *		Convert from a clock name to a clock pointer.
 */
clock_t
port_name_to_clock(mach_port_name_t clock_name)
{
	clock_t         clock = CLOCK_NULL;
	ipc_space_t     space;
	ipc_port_t      port;

	if (clock_name == 0) {
		return clock;
	}
	space = current_space();
	if (ipc_port_translate_send(space, clock_name, &port) != KERN_SUCCESS) {
		return clock;
	}
	clock = (clock_t)ipc_kobject_get_stable(port, IKOT_CLOCK);
	ip_mq_unlock(port);
	return clock;
}
