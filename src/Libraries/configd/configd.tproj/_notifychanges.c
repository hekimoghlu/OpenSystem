/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
__SCDynamicStoreCopyNotifiedKeys(SCDynamicStoreRef store, CFArrayRef *notifierKeys)
{
	serverSessionRef		mySession;
	SCDynamicStorePrivateRef	storePrivate = (SCDynamicStorePrivateRef)store;

	mySession = getSession(storePrivate->server);
	if (mySession->changedKeys != NULL) {
		*notifierKeys = mySession->changedKeys;
		mySession->changedKeys = NULL;
	} else {
		*notifierKeys = CFArrayCreate(NULL, NULL, 0, &kCFTypeArrayCallBacks);
	}

	return kSCStatusOK;
}


__private_extern__
kern_return_t
_notifychanges(mach_port_t			server,
	       xmlDataOut_t			*listRef,	/* raw XML bytes */
	       mach_msg_type_number_t		*listLen,
	       int				*sc_status
)
{
	CFIndex			len;
	serverSessionRef	mySession = getSession(server);
	CFArrayRef		notifierKeys;	/* array of CFStringRef's */
	Boolean			ok;

	*listRef = NULL;
	*listLen = 0;

	if (mySession == NULL) {
		*sc_status = kSCStatusNoStoreSession;	/* you must have an open session to play */
		return KERN_SUCCESS;
	}

	*sc_status = __SCDynamicStoreCopyNotifiedKeys(mySession->store, &notifierKeys);
	if (*sc_status != kSCStatusOK) {
		return KERN_SUCCESS;
	}

	/* serialize the array of keys */
	ok = _SCSerialize(notifierKeys, NULL, listRef, &len);
	*listLen = (mach_msg_type_number_t)len;
	CFRelease(notifierKeys);
	if (!ok) {
		*sc_status = kSCStatusFailed;
		return KERN_SUCCESS;
	}

	return KERN_SUCCESS;
}
