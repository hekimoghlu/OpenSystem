/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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

#include <unistd.h>

#include "configd.h"
#include "session.h"

static Boolean
isMySessionKey(CFStringRef sessionKey, CFStringRef key)
{
	CFDictionaryRef	dict;
	CFStringRef	storeSessionKey;

	dict = CFDictionaryGetValue(storeData, key);
	if (!dict) {
		/* if key no longer exists */
		return FALSE;
	}

	storeSessionKey = CFDictionaryGetValue(dict, kSCDSession);
	if (!storeSessionKey) {
		/* if this is not a session key */
		return FALSE;
	}

	if (!CFEqual(sessionKey, storeSessionKey)) {
		/* if this is not "my" session key */
		return FALSE;
	}

	return TRUE;
}


static void
removeAllKeys(SCDynamicStoreRef store, Boolean isRegex)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	CFArrayRef			keys;
	CFIndex				n;

	keys = isRegex ? storePrivate->patterns : storePrivate->keys;
	n = (keys != NULL) ? CFArrayGetCount(keys) : 0;
	if (n > 0) {
		CFIndex		i;
		CFArrayRef	keysToRemove;

		keysToRemove = CFArrayCreateCopy(NULL, keys);
		for (i = 0; i < n; i++) {
			(void) __SCDynamicStoreRemoveWatchedKey(store,
								CFArrayGetValueAtIndex(keysToRemove, i),
								isRegex,
								TRUE);
		}
		CFRelease(keysToRemove);
	}

	return;
}


__private_extern__
int
__SCDynamicStoreClose(SCDynamicStoreRef *store)
{
	serverSessionRef		mySession;
	SCDynamicStorePrivateRef	storePrivate = (SCDynamicStorePrivateRef)*store;

	SC_trace("close   : %5u",
		 storePrivate->server);

	/* Remove all notification keys and patterns */
	removeAllKeys(*store, FALSE);	// keys
	removeAllKeys(*store, TRUE);	// patterns

	/* Remove/cancel any outstanding notification requests. */
	__MACH_PORT_DEBUG(storePrivate->notifyPort != MACH_PORT_NULL, "*** __SCDynamicStoreClose", storePrivate->notifyPort);
	(void) __SCDynamicStoreNotifyCancel(*store);

	/* Remove any session keys */
	mySession = getSession(storePrivate->server);
	if (mySession->sessionKeys != NULL) {
		CFIndex		n	= CFArrayGetCount(mySession->sessionKeys);
		Boolean		push	= FALSE;
		CFStringRef	sessionKey;

		sessionKey = CFStringCreateWithFormat(NULL, NULL, CFSTR("%u"), storePrivate->server);
		for (CFIndex i = 0; i < n; i++) {
			CFStringRef	key	= CFArrayGetValueAtIndex(mySession->sessionKeys, i);

			if (isMySessionKey(sessionKey, key)) {
				(void) __SCDynamicStoreRemoveValue(*store, key, TRUE);
				push = TRUE;
			}
		}
		CFRelease(sessionKey);

		if (push) {
			/* push changes */
			(void) __SCDynamicStorePush();
		}
	}

	storePrivate->server = MACH_PORT_NULL;

	CFRelease(*store);
	*store = NULL;

	return kSCStatusOK;
}
