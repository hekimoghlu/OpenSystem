/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

Boolean
SCDynamicStoreNotifyCancel(SCDynamicStoreRef store)
{
	SCDynamicStorePrivateRef		storePrivate = (SCDynamicStorePrivateRef)store;
	kern_return_t				status;
	int					sc_status;

	if (!isA_SCDynamicStore(store)) {
		/* sorry, you must provide a session */
		_SCErrorSet(kSCStatusNoStoreSession);
		return FALSE;
	}

	switch (storePrivate->notifyStatus) {
		case NotifierNotRegistered :
			/* if no notifications have been registered */
			return TRUE;
		case Using_NotifierInformViaRunLoop :
			if (storePrivate->rls != NULL) {
				CFRunLoopSourceRef	rls;

				rls = storePrivate->rls;
				storePrivate->rls = NULL;

				CFRunLoopSourceInvalidate(rls);
				CFRelease(rls);
			}
			return TRUE;
		case Using_NotifierInformViaDispatch :
			(void) SCDynamicStoreSetDispatchQueue(store, NULL);
			return TRUE;
		default :
			break;
	}

	if (storePrivate->server == MACH_PORT_NULL) {
		/* sorry, you must have an open session to play */
		sc_status = kSCStatusNoStoreServer;
		goto done;
	}

	status = notifycancel(storePrivate->server, (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreNotifyCancel notifycancel()")) {
		sc_status = kSCStatusOK;
	}

    done :

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}
