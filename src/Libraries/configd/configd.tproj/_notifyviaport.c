/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
 * March 24, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include "configd.h"
#include "session.h"

__private_extern__
int
__SCDynamicStoreNotifyMachPort(SCDynamicStoreRef	store,
			       mach_msg_id_t		identifier,
			       mach_port_t		port)
{
	serverSessionRef		mySession;
	SCDynamicStorePrivateRef	storePrivate = (SCDynamicStorePrivateRef)store;

	if (storePrivate->notifyStatus != NotifierNotRegistered) {
		/* sorry, you can only have one notification registered at once */
		return kSCStatusNotifierActive;
	}

	if (identifier != 0) {
		/* sorry, the message ID (never used, no longer supported) must be zero */
		return kSCStatusInvalidArgument;
	}

	if (port == MACH_PORT_NULL) {
		/* sorry, you must specify a valid mach port */
		return kSCStatusInvalidArgument;
	}

	/* push out a notification if any changes are pending */
	mySession = getSession(storePrivate->server);
	if (mySession->changedKeys != NULL) {
		CFNumberRef	sessionNum;

		if (needsNotification == NULL)
			needsNotification = CFSetCreateMutable(NULL,
							       0,
							       &kCFTypeSetCallBacks);

		sessionNum = CFNumberCreate(NULL, kCFNumberIntType, &storePrivate->server);
		CFSetAddValue(needsNotification, sessionNum);
		CFRelease(sessionNum);
	}

	return kSCStatusOK;
}


__private_extern__
kern_return_t
_notifyviaport(mach_port_t	server,
	       mach_port_t	port,
	       mach_msg_id_t	identifier,
	       int		*sc_status
)
{
	serverSessionRef		mySession	= getSession(server);
	SCDynamicStorePrivateRef	storePrivate;

	if (mySession == NULL) {
		/* sorry, you must have an open session to play */
		*sc_status = kSCStatusNoStoreSession;
		if (port != MACH_PORT_NULL) {
			(void) mach_port_deallocate(mach_task_self(), port);
		}
		return KERN_SUCCESS;
	}
	storePrivate = (SCDynamicStorePrivateRef)mySession->store;

	*sc_status = __SCDynamicStoreNotifyMachPort(mySession->store, identifier, port);
	if (*sc_status != kSCStatusOK) {
		// if we can't enable the notification, release the provided callback port
		if (port != MACH_PORT_NULL) {
			__MACH_PORT_DEBUG(TRUE, "*** _notifyviaport __SCDynamicStoreNotifyMachPort failed: releasing port", port);
			(void) mach_port_deallocate(mach_task_self(), port);
		}
		return KERN_SUCCESS;
	}

	/* save notification port, requested identifier, and set notifier active */
	__MACH_PORT_DEBUG(TRUE, "*** _notifyviaport", port);
	storePrivate->notifyStatus         = Using_NotifierInformViaMachPort;
	storePrivate->notifyPort           = port;

	return KERN_SUCCESS;
}
