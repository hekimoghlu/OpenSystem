/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
 * - initial revision
 */

#include "SCDynamicStoreInternal.h"
#include "config.h"		/* MiG generated file */

Boolean
SCDynamicStoreSetNotificationKeys(SCDynamicStoreRef	store,
				  CFArrayRef		keys,
				  CFArrayRef		patterns)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	kern_return_t			status;
	CFDataRef			xmlKeys		= NULL;	/* keys (XML serialized) */
	xmlData_t			myKeysRef	= NULL;	/* keys (serialized) */
	CFIndex				myKeysLen	= 0;
	CFDataRef			xmlPatterns	= NULL;	/* patterns (XML serialized) */
	xmlData_t			myPatternsRef	= NULL;	/* patterns (serialized) */
	CFIndex				myPatternsLen	= 0;
	int				sc_status;
	CFMutableArrayRef		tmp;

	if (!__SCDynamicStoreNormalize(&store, FALSE)) {
		return FALSE;
	}

	/* serialize the keys */
	if (keys != NULL) {
		if (!_SCSerialize(keys, &xmlKeys, &myKeysRef, &myKeysLen)) {
			_SCErrorSet(kSCStatusFailed);
			return FALSE;
		}
	}

	/* serialize the patterns */
	if (patterns != NULL) {
		if (!_SCSerialize(patterns, &xmlPatterns, &myPatternsRef, &myPatternsLen)) {
			if (xmlKeys != NULL) CFRelease(xmlKeys);
			_SCErrorSet(kSCStatusFailed);
			return FALSE;
		}
	}

    retry :

	/* send the keys and patterns, fetch the associated result from the server */
	status = notifyset(storePrivate->server,
			   myKeysRef,
			   (mach_msg_type_number_t)myKeysLen,
			   myPatternsRef,
			   (mach_msg_type_number_t)myPatternsLen,
			   (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreSetNotificationKeys notifyset()")) {
		goto retry;
	}

	/* clean up */
	if (xmlKeys != NULL)		CFRelease(xmlKeys);
	if (xmlPatterns != NULL)	CFRelease(xmlPatterns);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	/* in case we need to re-connect, save the keys/patterns */
	tmp = (keys != NULL) ? CFArrayCreateMutableCopy(NULL, 0, keys) : NULL;
	if (storePrivate->keys != NULL) CFRelease(storePrivate->keys);
	storePrivate->keys = tmp;

	tmp = (patterns != NULL) ? CFArrayCreateMutableCopy(NULL, 0, patterns) : NULL;
	if (storePrivate->patterns != NULL) CFRelease(storePrivate->patterns);
	storePrivate->patterns = tmp;

	return TRUE;
}
