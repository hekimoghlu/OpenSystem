/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
 * April 5, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <unistd.h>
#include <sys/fileport.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "configd.h"
#include "session.h"


__private_extern__
int
__SCDynamicStoreNotifyFileDescriptor(SCDynamicStoreRef	store)
{
	serverSessionRef		mySession;
	SCDynamicStorePrivateRef	storePrivate = (SCDynamicStorePrivateRef)store;

	if (storePrivate->notifyStatus != NotifierNotRegistered) {
		/* sorry, you can only have one notification registered at once */
		return kSCStatusNotifierActive;
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
_notifyviafd(mach_port_t	server,
	     fileport_t		fileport,
	     int		identifier,
	     int		*sc_status
)
{
	int				fd		= -1;
	int				flags;
	serverSessionRef		mySession       = getSession(server);
	SCDynamicStorePrivateRef	storePrivate;

	/* get notification file descriptor */
	fd = fileport_makefd(fileport);
	mach_port_deallocate(mach_task_self(), fileport);
	if (fd < 0) {
		*sc_status = errno;
		return KERN_SUCCESS;
	}

	flags = fcntl(fd, F_GETFL, 0);
	if (flags == -1) {
		*sc_status = errno;
		goto fail;
	}

	flags |= O_NONBLOCK;
	if (fcntl(fd, F_SETFL, flags) == -1) {
		*sc_status = errno;
		goto fail;
	}

	if (mySession == NULL) {
		*sc_status = kSCStatusNoStoreSession;	/* you must have an open session to play */
		goto fail;
	}
	storePrivate = (SCDynamicStorePrivateRef)mySession->store;

	/* do common sanity checks */
	*sc_status = __SCDynamicStoreNotifyFileDescriptor(mySession->store);

	/* check status of __SCDynamicStoreNotifyFileDescriptor() */
	if (*sc_status != kSCStatusOK) {
		goto fail;
	}

	/* set notifier active */
	storePrivate->notifyStatus         = Using_NotifierInformViaFD;
	storePrivate->notifyFile           = fd;
	storePrivate->notifyFileIdentifier = identifier;

	return KERN_SUCCESS;

    fail :

	if (fd >= 0) close(fd);
	return KERN_SUCCESS;
}
