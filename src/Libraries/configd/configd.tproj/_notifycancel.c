/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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

#include <unistd.h>

#include "configd.h"
#include "session.h"


__private_extern__
int
__SCDynamicStoreNotifyCancel(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef	storePrivate = (SCDynamicStorePrivateRef)store;

	/*
	 * cleanup any mach port based notifications.
	 */
	if (storePrivate->notifyPort != MACH_PORT_NULL) {
		__MACH_PORT_DEBUG(TRUE, "*** __SCDynamicStoreNotifyCancel (notify port)", storePrivate->notifyPort);
		(void) mach_port_deallocate(mach_task_self(), storePrivate->notifyPort);
		storePrivate->notifyPort = MACH_PORT_NULL;
	}

	/*
	 * cleanup any file based notifications.
	 */
	if (storePrivate->notifyFile != -1) {
		// close (notification) fd
		(void) close(storePrivate->notifyFile);
		storePrivate->notifyFile = -1;
	}

	/* remove this session from the to-be-notified list */
	if (needsNotification) {
		CFNumberRef	num;

		num = CFNumberCreate(NULL, kCFNumberIntType, &storePrivate->server);
		CFSetRemoveValue(needsNotification, num);
		CFRelease(num);

		if (CFSetGetCount(needsNotification) == 0) {
			CFRelease(needsNotification);
			needsNotification = NULL;
		}
	}

	/* set notifier inactive */
	storePrivate->notifyStatus = NotifierNotRegistered;

	return kSCStatusOK;
}


__private_extern__
kern_return_t
_notifycancel(mach_port_t	server,
	      int		*sc_status)
{
	serverSessionRef	mySession = getSession(server);

	if (mySession == NULL) {
		*sc_status = kSCStatusNoStoreSession;	/* you must have an open session to play */
		return KERN_SUCCESS;
	}

	__MACH_PORT_DEBUG(((SCDynamicStorePrivateRef)mySession->store)->notifyPort != MACH_PORT_NULL,
			  "*** _notifycancel",
			  ((SCDynamicStorePrivateRef)mySession->store)->notifyPort);
	*sc_status = __SCDynamicStoreNotifyCancel(mySession->store);
	return KERN_SUCCESS;
}
