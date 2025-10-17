/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
SCDynamicStoreSetMultiple(SCDynamicStoreRef	store,
			  CFDictionaryRef	keysToSet,
			  CFArrayRef		keysToRemove,
			  CFArrayRef		keysToNotify)
{
	SCDynamicStorePrivateRef	storePrivate;
	kern_return_t			status;
	CFDataRef			xmlSet		= NULL;	/* key/value pairs to set (XML serialized) */
	xmlData_t			mySetRef	= NULL;	/* key/value pairs to set (serialized) */
	CFIndex				mySetLen	= 0;
	CFDataRef			xmlRemove	= NULL;	/* keys to remove (XML serialized) */
	xmlData_t			myRemoveRef	= NULL;	/* keys to remove (serialized) */
	CFIndex				myRemoveLen	= 0;
	CFDataRef			xmlNotify	= NULL;	/* keys to notify (XML serialized) */
	xmlData_t			myNotifyRef	= NULL;	/* keys to notify (serialized) */
	CFIndex				myNotifyLen	= 0;
	int				sc_status;

	if (!__SCDynamicStoreNormalize(&store, TRUE)) {
		return FALSE;
	}

	/* serialize the key/value pairs to set*/
	if (keysToSet != NULL) {
		CFDictionaryRef	newInfo;
		Boolean		ok;

		newInfo = _SCSerializeMultiple(keysToSet);
		if (newInfo == NULL) {
			_SCErrorSet(kSCStatusInvalidArgument);
			return FALSE;
		}

		ok = _SCSerialize(newInfo, &xmlSet, &mySetRef, &mySetLen);
		CFRelease(newInfo);
		if (!ok) {
			_SCErrorSet(kSCStatusInvalidArgument);
			return FALSE;
		}
	}

	/* serialize the keys to remove */
	if (keysToRemove != NULL) {
		if (!_SCSerialize(keysToRemove, &xmlRemove, &myRemoveRef, &myRemoveLen)) {
			if (xmlSet != NULL)	CFRelease(xmlSet);
			_SCErrorSet(kSCStatusInvalidArgument);
			return FALSE;
		}
	}

	/* serialize the keys to notify */
	if (keysToNotify != NULL) {
		if (!_SCSerialize(keysToNotify, &xmlNotify, &myNotifyRef, &myNotifyLen)) {
			if (xmlSet != NULL)	CFRelease(xmlSet);
			if (xmlRemove != NULL)	CFRelease(xmlRemove);
			_SCErrorSet(kSCStatusInvalidArgument);
			return FALSE;
		}
	}

	storePrivate = (SCDynamicStorePrivateRef)store;

    retry :

	/* send the keys and patterns, fetch the associated result from the server */
	status = configset_m(storePrivate->server,
			     mySetRef,
			     (mach_msg_type_number_t)mySetLen,
			     myRemoveRef,
			     (mach_msg_type_number_t)myRemoveLen,
			     myNotifyRef,
			     (mach_msg_type_number_t)myNotifyLen,
			     (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreSetMultiple configset_m()")) {
		goto retry;
	}

	/* clean up */
	if (xmlSet != NULL)	CFRelease(xmlSet);
	if (xmlRemove != NULL)	CFRelease(xmlRemove);
	if (xmlNotify != NULL)	CFRelease(xmlNotify);

	sc_status = __SCDynamicStoreMapInternalStatus(sc_status, TRUE);

	if (sc_status != kSCStatusOK) {
		_SCErrorSet(sc_status);
		return FALSE;
	}

	return TRUE;
}

Boolean
SCDynamicStoreSetValue(SCDynamicStoreRef store, CFStringRef key, CFPropertyListRef value)
{
	SCDynamicStorePrivateRef	storePrivate;
	kern_return_t			status;
	CFDataRef			utfKey;		/* serialized key */
	xmlData_t			myKeyRef;
	CFIndex				myKeyLen;
	CFDataRef			xmlData;	/* serialized data */
	xmlData_t			myDataRef;
	CFIndex				myDataLen;
	int				sc_status;
	int				newInstance;

	if (!__SCDynamicStoreNormalize(&store, TRUE)) {
		return FALSE;
	}

	storePrivate = (SCDynamicStorePrivateRef)store;

	if (storePrivate->cache_active) {
		if (storePrivate->cached_removals != NULL) {
			CFIndex	i;

			i = CFArrayGetFirstIndexOfValue(storePrivate->cached_removals,
							CFRangeMake(0, CFArrayGetCount(storePrivate->cached_removals)),
							key);
			if (i != kCFNotFound) {
				// if previously "removed"
				CFArrayRemoveValueAtIndex(storePrivate->cached_removals, i);
			}
		}

		if (storePrivate->cached_set == NULL) {
			storePrivate->cached_set = CFDictionaryCreateMutable(NULL,
									     0,
									     &kCFTypeDictionaryKeyCallBacks,
									     &kCFTypeDictionaryValueCallBacks);
		}
		CFDictionarySetValue(storePrivate->cached_set, key, value);
		return TRUE;
	}

	/* serialize the key */
	if (!_SCSerializeString(key, &utfKey, &myKeyRef, &myKeyLen)) {
		_SCErrorSet(kSCStatusInvalidArgument);
		return FALSE;
	}

	/* serialize the data */
	if (!_SCSerialize(value, &xmlData, &myDataRef, &myDataLen)) {
		CFRelease(utfKey);
		_SCErrorSet(kSCStatusInvalidArgument);
		return FALSE;
	}

    retry :

	/* send the key & data to the server, get new instance id */
	status = configset(storePrivate->server,
			   myKeyRef,
			   (mach_msg_type_number_t)myKeyLen,
			   myDataRef,
			   (mach_msg_type_number_t)myDataLen,
			   0,
			   &newInstance,
			   (int *)&sc_status);

	if (__SCDynamicStoreCheckRetryAndHandleError(store,
						     status,
						     &sc_status,
						     "SCDynamicStoreSetValue configset()")) {
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
