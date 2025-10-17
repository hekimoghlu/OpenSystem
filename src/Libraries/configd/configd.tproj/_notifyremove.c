/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include "pattern.h"


static int
removeKey(CFMutableArrayRef keys, CFStringRef key)
{
	CFIndex	i;
	CFIndex	n;

	if (keys == NULL) {
		/* sorry, empty notifier list */
		return kSCStatusNoKey;
	}

	n = CFArrayGetCount(keys);
	i = CFArrayGetFirstIndexOfValue(keys, CFRangeMake(0, n), key);
	if (i == kCFNotFound) {
		/* sorry, key does not exist in notifier list */
		return kSCStatusNoKey;
	}

	/* remove key from this sessions notifier list */
	CFArrayRemoveValueAtIndex(keys, i);
	return kSCStatusOK;
}


__private_extern__
int
__SCDynamicStoreRemoveWatchedKey(SCDynamicStoreRef store, CFStringRef key, Boolean isRegex, Boolean internal)
{
	int				sc_status	= kSCStatusOK;
	CFNumberRef			sessionNum;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;

	SC_trace("%s : %5u : %s : %@",
		 internal ? "*watch-" : "watch- ",
		 storePrivate->server,
		 isRegex  ? "pattern" : "key",
		 key);

	/*
	 * remove key from this sessions notifier list after checking that
	 * it was previously defined.
	 */
	if (isRegex) {
		sc_status = removeKey(storePrivate->patterns, key);
		if (sc_status != kSCStatusOK) {
			goto done;
		}

		/* remove this session as a pattern watcher */
		sessionNum = CFNumberCreate(NULL, kCFNumberIntType, &storePrivate->server);
		patternRemoveSession(key, sessionNum);
		CFRelease(sessionNum);
	} else {
		sc_status = removeKey(storePrivate->keys, key);
		if (sc_status != kSCStatusOK) {
			goto done;
		}

		/*
		 * We are watching a specific key. As such, update the
		 * store to remove our interest in any changes.
		 */
		sessionNum = CFNumberCreate(NULL, kCFNumberIntType, &storePrivate->server);
		_storeRemoveWatcher(sessionNum, key);
		CFRelease(sessionNum);
	}

    done :

	return sc_status;
}


__private_extern__
kern_return_t
_notifyremove(mach_port_t		server,
	      xmlData_t			keyRef,		/* raw XML bytes */
	      mach_msg_type_number_t	keyLen,
	      int			isRegex,
	      int			*sc_status
)
{
	CFStringRef		key		= NULL;		/* key  (un-serialized) */
	serverSessionRef	mySession;

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
		*sc_status = kSCStatusNoStoreSession;	/* you must have an open session to play */
		goto done;
	}

	*sc_status = __SCDynamicStoreRemoveWatchedKey(mySession->store,
						      key,
						      isRegex != 0,
						      FALSE);

    done :

	if (key)	CFRelease(key);
	return KERN_SUCCESS;
}

