/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

#include "SCDynamicStoreInternal.h"
#include "config.h"		/* MiG generated file */

#include <paths.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

Boolean
SCDynamicStoreNotifyFileDescriptor(SCDynamicStoreRef	store,
				    int32_t		identifier,
				    int			*fd)
{
	int					fildes[2]	= { -1, -1 };
	fileport_t				fileport	= MACH_PORT_NULL;
	int					ret;
	int					sc_status;
	SCDynamicStorePrivateRef		storePrivate	= (SCDynamicStorePrivateRef)store;
	kern_return_t				status;

	if (!__SCDynamicStoreNormalize(&store, FALSE)) {
		return FALSE;
	}

	if (storePrivate->notifyStatus != NotifierNotRegistered) {
		/* sorry, you can only have one notification registered at once */
		_SCErrorSet(kSCStatusNotifierActive);
		return FALSE;
	}

	ret = pipe(fildes);
	if (ret == -1) {
		_SCErrorSet(errno);
		SC_log(LOG_ERR, "pipe() failed: %s", strerror(errno));
		goto fail;
	}

	/*
	 * send fildes[1], the sender's fd, to configd using a fileport and
	 * return fildes[0] to the caller.
	 */

	fileport = MACH_PORT_NULL;
	ret = fileport_makeport(fildes[1], &fileport);
	if (ret < 0) {
		_SCErrorSet(errno);
		SC_log(LOG_ERR, "fileport_makeport() failed: %s", strerror(errno));
		goto fail;
	}

    retry :

	status = notifyviafd(storePrivate->server,
			     fileport,
			     identifier,
			     (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreNotifyFileDescriptor notifyviafd()")) {
		goto retry;
	}

	if (status != KERN_SUCCESS) {
		_SCErrorSet(status);
		goto fail;
	}

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		goto fail;
	}

	/* the SCDynamicStore server now has a copy of the write side, close our reference */
	(void) close(fildes[1]);

	/* and keep the read side */
	*fd = fildes[0];

	/* set notifier active */
	storePrivate->notifyStatus = Using_NotifierInformViaFD;

	return TRUE;

    fail :

	if (fildes[0] != -1) close(fildes[0]);
	if (fildes[1] != -1) close(fildes[1]);
	return FALSE;
}
