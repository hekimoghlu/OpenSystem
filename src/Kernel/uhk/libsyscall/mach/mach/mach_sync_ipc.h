/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

#ifndef _MACH_SYNC_IPC_H_
#define _MACH_SYNC_IPC_H_

#include <mach/mach.h>

__BEGIN_DECLS

/*!
 * @function mach_sync_ipc_link_monitoring_start
 *
 * @abstract
 * Starts monitoring the sync IPC priority inversion avoidance
 * facility of the current thread.
 * A subsequent call to mach_sync_ipc_link_monitoring_stop() will
 * validate that the facility took effect for all synchronous IPC
 * performed from this thread between the calls to start and stop.
 *
 * @discussion
 * In case of success, a port right is returned, which has to be
 * deallocated by passing it to mach_sync_ipc_link_monitoring_stop().
 *
 * @param port
 * Pointer to a mach_port_t that will be populated in case of success.
 *
 * @result
 * KERN_SUCCESS in case of success, specific error otherwise.
 * If the call is not supported, KERN_NOT_SUPPORTED is returned.
 */
extern kern_return_t mach_sync_ipc_link_monitoring_start(mach_port_t* port);

/*!
 * @function mach_sync_ipc_link_monitoring_stop
 *
 * @abstract
 * Stops monitoring the sync IPC priority inversion avoidance facility
 * of the current thread started by a call to mach_sync_ipc_link_monitoring_start().
 *
 * Returns whether the facility took effect for all synchronous IPC performed
 * from this thread between the calls to start and stop.
 *
 * Reasons for this function to return false include:
 * -remote message event handler did not reply to the message itself
 * -remote message was not received by a workloop (xpc connection or dispatch mach channel)
 *
 * @discussion
 * To be called after mach_sync_ipc_link_monitoring_start(). If
 * mach_sync_ipc_link_monitoring_start() didn't return an error this
 * function must be called to deallocate the port right that was returned.
 *
 * @param port
 * mach_port_t returned by mach_sync_ipc_link_monitoring_start().
 *
 * @param in_effect
 * Pointer to boolean_t value that will be populated in the case of success.
 * Indicates whether the sync IPC priority inversion avoidance facility took
 * effect for all synchronous IPC performed from this thread between the calls
 * to start and stop.
 *
 * @result
 * KERN_SUCCESS in case of no errors, specific error otherwise.
 * If the call is not supported, KERN_NOT_SUPPORTED is returned.
 */
extern kern_return_t mach_sync_ipc_link_monitoring_stop(mach_port_t port, boolean_t* in_effect);

typedef enum thread_destruct_special_reply_port_rights {
	THREAD_SPECIAL_REPLY_PORT_ALL,
	THREAD_SPECIAL_REPLY_PORT_RECEIVE_ONLY,
	THREAD_SPECIAL_REPLY_PORT_SEND_ONLY,
} thread_destruct_special_reply_port_rights_t;

extern kern_return_t thread_destruct_special_reply_port(mach_port_name_t port, thread_destruct_special_reply_port_rights_t rights);

extern mach_port_t mig_get_special_reply_port(void);

extern void mig_dealloc_special_reply_port(mach_port_t migport);


__END_DECLS

#endif  /* _MACH_SYNC_IPC_H_ */
