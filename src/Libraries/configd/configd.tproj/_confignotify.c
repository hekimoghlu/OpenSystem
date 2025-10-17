/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
 * May 19, 2001			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include "configd.h"
#include "session.h"

__private_extern__
int
__SCDynamicStoreNotifyValue(SCDynamicStoreRef store, CFStringRef key, Boolean internal)
{
	CFDictionaryRef			dict;
	Boolean				newValue	= FALSE;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	int				sc_status	= kSCStatusOK;
	CFDataRef			value;

	SC_trace("%s : %5u : %@",
		 internal ? "*notify" : "notify ",
		 storePrivate->server,
		 key);

	/*
	 * Tickle the value in the dynamic store
	 */
	dict = CFDictionaryGetValue(storeData, key);
	if (!dict || !CFDictionaryGetValueIfPresent(dict, kSCDData, (const void **)&value)) {
		/* key doesn't exist (or data never defined) */
		(void)_SCSerialize(kCFBooleanTrue, &value, NULL, NULL);
		newValue = TRUE;
	}

	/* replace or store initial/temporary existing value */
	__SCDynamicStoreSetValue(store, key, value, TRUE);

	if (newValue) {
		/* remove the value we just created */
		__SCDynamicStoreRemoveValue(store, key, TRUE);
		CFRelease(value);
	}

	if (!internal) {
		/* push changes */
		__SCDynamicStorePush();
	}

	return sc_status;
}


__private_extern__
kern_return_t
_confignotify(mach_port_t 		server,
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

		SC_trace("!notify : %5u : %@",
			 storePrivate->server,
			 key);
#endif	// DEBUG
		*sc_status = status;
		goto done;
	}

	*sc_status = __SCDynamicStoreNotifyValue(mySession->store, key, FALSE);

    done :

	if (key)	CFRelease(key);
	return KERN_SUCCESS;
}
