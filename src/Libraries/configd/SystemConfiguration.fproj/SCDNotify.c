/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
 * May 19, 2001		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include "SCDynamicStoreInternal.h"
#include "config.h"		/* MiG generated file */

Boolean
SCDynamicStoreNotifyValue(SCDynamicStoreRef	store,
			  CFStringRef		key)
{
	SCDynamicStorePrivateRef	storePrivate;
	kern_return_t			status;
	CFDataRef			utfKey;		/* serialized key */
	xmlData_t			myKeyRef;
	CFIndex				myKeyLen;
	int				sc_status;

	if (!__SCDynamicStoreNormalize(&store, TRUE)) {
		return FALSE;
	}

	storePrivate = (SCDynamicStorePrivateRef)store;

	if (storePrivate->cache_active) {
		if (storePrivate->cached_notifys == NULL)  {
			storePrivate->cached_notifys = CFArrayCreateMutable(NULL,
									    0,
									    &kCFTypeArrayCallBacks);
		}

		if (!CFArrayContainsValue(storePrivate->cached_notifys,
					  CFRangeMake(0, CFArrayGetCount(storePrivate->cached_notifys)),
					  key)) {
			CFArrayAppendValue(storePrivate->cached_notifys, key);
		}

		return TRUE;
	}

	/* serialize the key */
	if (!_SCSerializeString(key, &utfKey, &myKeyRef, &myKeyLen)) {
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

    retry :

	/* send the key to the server */
	status = confignotify(storePrivate->server,
			      myKeyRef,
			      (mach_msg_type_number_t)myKeyLen,
			      (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreNotifyValue confignotify()")) {
		goto retry;
	}

	/* clean up */
	CFRelease(utfKey);

	sc_status = __SCDynamicStoreMapInternalStatus(sc_status, TRUE);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}
