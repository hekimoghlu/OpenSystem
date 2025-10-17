/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
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
 *	File:	ipc/ipc_notify.h
 *	Author:	Rich Draves
 *	Date:	1989
 *
 *	Declarations of notification-sending functions.
 */

#ifndef _IPC_IPC_NOTIFY_H_
#define _IPC_IPC_NOTIFY_H_

#include <mach/port.h>

#pragma GCC visibility push(hidden)

typedef struct ipc_notify_nsenders {
	ipc_port_t              ns_notify;
	mach_port_mscount_t     ns_mscount;
	boolean_t               ns_is_kobject;
} ipc_notify_nsenders_t;

/*
 * Exported interfaces
 */

/* Send a port-deleted notification */
extern void ipc_notify_port_deleted(
	ipc_port_t              port,
	mach_port_name_t        name);

/* Send a send-possible notification */
extern void ipc_notify_send_possible(
	ipc_port_t              port,
	mach_port_name_t        name);

/* Send a port-destroyed notification */
extern void ipc_notify_port_destroyed(
	ipc_port_t              port,
	ipc_port_t              right);

/* Send a no-senders notification */
extern void ipc_notify_no_senders(
	ipc_port_t              notify,
	mach_port_mscount_t     mscount,
	boolean_t               kobject);

extern ipc_notify_nsenders_t ipc_notify_no_senders_prepare(
	ipc_port_t              port);

static inline void
ipc_notify_no_senders_emit(ipc_notify_nsenders_t nsrequest)
{
	if (nsrequest.ns_notify) {
		ipc_notify_no_senders(nsrequest.ns_notify,
		    nsrequest.ns_mscount, nsrequest.ns_is_kobject);
	}
}

extern void ipc_notify_no_senders_consume(
	ipc_notify_nsenders_t   nsrequest);

/* Send a send-once notification */
extern void ipc_notify_send_once_and_unlock(
	ipc_port_t              port);

/* Send a dead-name notification */
extern void ipc_notify_dead_name(
	ipc_port_t              port,
	mach_port_name_t        name);

#pragma GCC visibility pop

#endif  /* _IPC_IPC_NOTIFY_H_ */
