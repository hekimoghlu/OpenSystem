/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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

#include "SCDynamicStoreInternal.h"
#include "config.h"		/* MiG generated file */

Boolean
SCDynamicStoreAddTemporaryValue(SCDynamicStoreRef store, CFStringRef key, CFPropertyListRef value)
{
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	kern_return_t			status;
	CFDataRef			utfKey;		/* serialized key */
	xmlData_t			myKeyRef;
	CFIndex				myKeyLen;
	CFDataRef			xmlData;	/* serialized data */
	xmlData_t			myDataRef;
	CFIndex				myDataLen;
	int				newInstance;
	int				sc_status;

	if (!__SCDynamicStoreNormalize(&store, FALSE)) {
		return FALSE;
	}

	/* serialize the key */
	if (!_SCSerializeString(key, &utfKey, &myKeyRef, &myKeyLen)) {
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

	/* serialize the data */
	if (!_SCSerialize(value, &xmlData, &myDataRef, &myDataLen)) {
		CFRelease(utfKey);
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

    retry :

	/* send the key & data to the server */
	status = configadd_s(storePrivate->server,
			     myKeyRef,
			     (mach_msg_type_number_t)myKeyLen,
			     myDataRef,
			     (mach_msg_type_number_t)myDataLen,
			     &newInstance,
			     (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreAddTemporaryValue configadd_s()")) {
		goto retry;
	}

	/* clean up */
	CFRelease(utfKey);
	CFRelease(xmlData);

	sc_status = __SCDynamicStoreMapInternalStatus(sc_status, TRUE);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}

Boolean
SCDynamicStoreAddValue(SCDynamicStoreRef store, CFStringRef key, CFPropertyListRef value)
{
	SCDynamicStorePrivateRef	storePrivate;
	kern_return_t			status;
	CFDataRef			utfKey;		/* serialized key */
	xmlData_t			myKeyRef;
	CFIndex				myKeyLen;
	CFDataRef			xmlData;	/* serialized data */
	xmlData_t			myDataRef;
	CFIndex				myDataLen;
	int				newInstance;
	int				sc_status;

	if (!__SCDynamicStoreNormalize(&store, TRUE)) {
		return FALSE;
	}

	/* serialize the key */
	if (!_SCSerializeString(key, &utfKey, &myKeyRef, &myKeyLen)) {
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

	/* serialize the data */
	if (!_SCSerialize(value, &xmlData, &myDataRef, &myDataLen)) {
		CFRelease(utfKey);
		_SCErrorSet(kSCStatusFailed);
		return FALSE;
	}

	storePrivate = (SCDynamicStorePrivateRef)store;

    retry :

	/* send the key & data to the server */
	status = configadd(storePrivate->server,
			   myKeyRef,
			   (mach_msg_type_number_t)myKeyLen,
			   myDataRef,
			   (mach_msg_type_number_t)myDataLen,
			   &newInstance,
			   (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreAddValue configadd()")) {
		goto retry;
	}

	/* clean up */
	CFRelease(utfKey);
	CFRelease(xmlData);

	sc_status = __SCDynamicStoreMapInternalStatus(sc_status, TRUE);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}
