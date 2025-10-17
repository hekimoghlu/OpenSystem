/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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

#define	N_QUICK	64

__private_extern__
int
__SCDynamicStoreCopyKeyList(SCDynamicStoreRef store, CFStringRef key, Boolean isRegex, CFArrayRef *subKeys)
{
	CFMutableArrayRef		keyArray;
	CFIndex				storeCnt;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	CFStringRef			storeStr;
	CFDictionaryRef			storeValue;

	SC_trace("list    : %5u : %s : %@",
		 storePrivate->server,
		 isRegex  ? "pattern" : "key",
		 key);

	if (isRegex) {
		*subKeys = patternCopyMatches(key);
		return (*subKeys != NULL) ? kSCStatusOK : kSCStatusFailed;
	}

	storeCnt = CFDictionaryGetCount(storeData);
	keyArray = CFArrayCreateMutable(NULL, storeCnt, &kCFTypeArrayCallBacks);
	if (storeCnt > 0) {
		int		i;
		const void *	storeKeys_q[N_QUICK];
		const void **	storeKeys	= storeKeys_q;
		const void *	storeValues_q[N_QUICK];
		const void **	storeValues	= storeValues_q;

		if (storeCnt > (CFIndex)(sizeof(storeKeys_q) / sizeof(CFStringRef))) {
			storeKeys   = CFAllocatorAllocate(NULL, storeCnt * sizeof(CFStringRef), 0);
			storeValues = CFAllocatorAllocate(NULL, storeCnt * sizeof(CFStringRef), 0);
		}

		CFDictionaryGetKeysAndValues(storeData, storeKeys, storeValues);
		for (i = 0; i < storeCnt; i++) {
			storeStr   = (CFStringRef)storeKeys[i];
			storeValue = (CFDictionaryRef)storeValues[i];
			/*
			 * only return those keys which are prefixed by the
			 * provided key string and have data.
			 */
			if (((CFStringGetLength(key) == 0) || CFStringHasPrefix(storeStr, key)) &&
			    CFDictionaryContainsKey(storeValue, kSCDData)) {
				CFArrayAppendValue(keyArray, storeStr);
			}
		}

		if (storeKeys != storeKeys_q) {
			CFAllocatorDeallocate(NULL, storeKeys);
			CFAllocatorDeallocate(NULL, storeValues);
		}
	}

	*subKeys = CFArrayCreateCopy(NULL, keyArray);
	CFRelease(keyArray);
	return kSCStatusOK;
}


__private_extern__
kern_return_t
_configlist(mach_port_t			server,
	    xmlData_t			keyRef,		/* raw XML bytes */
	    mach_msg_type_number_t	keyLen,
	    int				isRegex,
	    xmlDataOut_t		*listRef,	/* raw XML bytes */
	    mach_msg_type_number_t	*listLen,
	    int				*sc_status)
{
	CFStringRef		key		= NULL;		/* key  (un-serialized) */
	CFIndex			len;
	serverSessionRef	mySession;
	Boolean			ok;
	CFArrayRef		subKeys;			/* array of CFStringRef's */

	*listRef = NULL;
	*listLen = 0;

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

	*sc_status = __SCDynamicStoreCopyKeyList(mySession->store, key, isRegex != 0, &subKeys);
	if (*sc_status != kSCStatusOK) {
		goto done;
	}

	/* serialize the list of keys */
	ok = _SCSerialize(subKeys, NULL, listRef, &len);
	*listLen = (mach_msg_type_number_t)len;
	CFRelease(subKeys);
	if (!ok) {
		*sc_status = kSCStatusFailed;
		goto done;
	}

    done :

	if (key)	CFRelease(key);
	return KERN_SUCCESS;
}
