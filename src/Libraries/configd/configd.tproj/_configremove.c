/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
__SCDynamicStoreRemoveValue(SCDynamicStoreRef store, CFStringRef key, Boolean internal)
{
	CFDictionaryRef			dict;
	CFMutableDictionaryRef		newDict;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	int				sc_status	= kSCStatusOK;
	CFStringRef			sessionKey;

	SC_trace("%s : %5u : %@",
		 internal ? "*remove" : "remove ",
		 storePrivate->server,
		 key);

	/*
	 * Ensure that this key exists.
	 */
	dict = CFDictionaryGetValue(storeData, key);
	if ((dict == NULL) || !CFDictionaryContainsKey(dict, kSCDData)) {
		/* key doesn't exist (or data never defined) */
		sc_status = kSCStatusNoKey;
		goto done;
	}
	newDict = CFDictionaryCreateMutableCopy(NULL, 0, dict);

	/*
	 * Mark this key as "changed". Any "watchers" will be
	 * notified as soon as the lock is released.
	 */
	CFSetAddValue(changedKeys, key);

	/*
	 * Add this key to a deferred cleanup list so that, after
	 * the change notifications are posted, any associated
	 * regex keys can be removed.
	 */
	CFSetAddValue(deferredRemovals, key);

	/*
	 * Check if this is a session key and, if so, add it
	 * to the (session) removal list
	 */
	sessionKey = CFDictionaryGetValue(newDict, kSCDSession);
	if (sessionKey) {
		CFStringRef	removedKey;

		/* We are no longer a session key! */
		CFDictionaryRemoveValue(newDict, kSCDSession);

		/* add this session key to the (session) removal list */
		removedKey = CFStringCreateWithFormat(NULL, 0, CFSTR("%@:%@"), sessionKey, key);
		CFSetAddValue(removedSessionKeys, removedKey);
		CFRelease(removedKey);
	}

	/*
	 * Remove data and update/remove the dictionary store entry.
	 */
	CFDictionaryRemoveValue(newDict, kSCDData);
	if (CFDictionaryGetCount(newDict) > 0) {
		/* this key is still being "watched" */
		CFDictionarySetValue(storeData, key, newDict);
	} else {
		/* no information left, remove the empty dictionary */
		CFDictionaryRemoveValue(storeData, key);
	}
	CFRelease(newDict);

	if (!internal) {
		/* push changes */
		__SCDynamicStorePush();
	}

    done:

	return sc_status;
}


__private_extern__
kern_return_t
_configremove(mach_port_t		server,
	      xmlData_t			keyRef,		/* raw XML bytes */
	      mach_msg_type_number_t	keyLen,
	      int			*sc_status)
{
	CFStringRef		key		= NULL;		/* key  (un-serialized) */
	serverSessionRef	mySession;
	int			status;

	/* un-serialize the key */
	if (!_SCUnserializeString(&key, NULL, keyRef, keyLen)) {
		*sc_status = kSCStatusFailed;
		goto done;
	}

	if (!isA_CFString(key)) {
		*sc_status = kSCStatusInvalidArgument;
		goto done;
	}

	mySession = getSession(server);
	if (mySession == NULL) {
		/* you must have an open session to play */
		*sc_status = kSCStatusNoStoreSession;
		goto done;
	}

	status = checkWriteAccess(mySession, key);
	if (status != kSCStatusOK) {
#ifdef	DEBUG
		SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)mySession->store;

		SC_trace("!remove : %5u : %@",
			 storePrivate->server,
			 key);
#endif	// DEBUG
		*sc_status = status;
		goto done;
	}

	*sc_status = __SCDynamicStoreRemoveValue(mySession->store, key, FALSE);

    done :

	if (key)	CFRelease(key);

	return KERN_SUCCESS;
}

