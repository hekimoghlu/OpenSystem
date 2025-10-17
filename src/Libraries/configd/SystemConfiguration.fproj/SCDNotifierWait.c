/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * March 31, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include "SCDynamicStoreInternal.h"
#include "config.h"		/* MiG generated file */
#include <mach/mach_error.h>


static mach_msg_id_t
waitForMachMessage(mach_port_t port)
{
	union {
		u_int8_t		buf[sizeof(mach_msg_empty_t) + MAX_TRAILER_SIZE];
		mach_msg_empty_rcv_t	msg;
	} notify_msg;
	kern_return_t 	status;

	status = mach_msg(&notify_msg.msg.header,	/* msg */
			  MACH_RCV_MSG,			/* options */
			  0,				/* send_size */
			  sizeof(notify_msg),		/* rcv_size */
			  port,				/* rcv_name */
			  MACH_MSG_TIMEOUT_NONE,	/* timeout */
			  MACH_PORT_NULL);		/* notify */
	if (status != KERN_SUCCESS) {
		SC_log(LOG_NOTICE, "mach_msg() failed: %s", mach_error_string(status));
		return -1;
	}

	return notify_msg.msg.header.msgh_id;
}


Boolean
SCDynamicStoreNotifyWait(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	kern_return_t			status;
	mach_port_t			port;
	int				sc_status;
	mach_msg_id_t			msgid;

	if (!__SCDynamicStoreNormalize(&store, FALSE)) {
		return FALSE;
	}

	if (storePrivate->notifyStatus != NotifierNotRegistered) {
		/* sorry, you can only have one notification registered at once */
		_SCErrorSet(kSCStatusNotifierActive);
		return FALSE;
	}

	port = __SCDynamicStoreAddNotificationPort(store);
	if (port == MACH_PORT_NULL) {
		return FALSE;
	}

	/* set notifier active */
	storePrivate->notifyStatus = Using_NotifierWait;

	msgid = waitForMachMessage(port);

	/* set notifier inactive */
	storePrivate->notifyStatus = NotifierNotRegistered;

	if (msgid == MACH_NOTIFY_NO_SENDERS) {
		/* the server closed the notifier port */
#ifdef	DEBUG
		SC_log(LOG_DEBUG, "notifier port closed, port %u", port);
#endif	/* DEBUG */
		_SCErrorSet(kSCStatusNoStoreServer);
		return FALSE;
	}

	if (msgid == -1) {
		/* one of the mach routines returned an error */
#ifdef	DEBUG
		SC_log(LOG_DEBUG, "communication with server failed, remove port right %u", port);
#endif	/* DEBUG */
		(void) mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_RECEIVE , -1);
		_SCErrorSet(kSCStatusNoStoreServer);
		return FALSE;
	}

	// something changed, cancelling notification request
	status = notifycancel(storePrivate->server, (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreNotifyWait notifycancel()")) {
		sc_status = kSCStatusOK;
	}

	/* remove our receive right  */
	__SCDynamicStoreRemoveNotificationPort(store, port);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}
