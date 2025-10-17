/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
__SCDynamicStoreAddValue(SCDynamicStoreRef store, CFStringRef key, CFDataRef value)
{
	int				sc_status	= kSCStatusOK;
	SCDynamicStorePrivateRef	storePrivate	= (SCDynamicStorePrivateRef)store;
	CFDataRef			tempValue;

	SC_trace("add  %s : %5u : %@",
		 storePrivate->useSessionKeys ? "t " : "  ",
		 storePrivate->server,
		 key);

	/*
	 * Ensure that this is a new key.
	 */
	sc_status = __SCDynamicStoreCopyValue(store, key, NULL, &tempValue, TRUE);
	switch (sc_status) {
		case kSCStatusNoKey :
			/* store key does not exist, proceed */
			break;

		case kSCStatusOK :
			/* store key exists, sorry */
			CFRelease(tempValue);
			sc_status = kSCStatusKeyExists;
			goto done;

		default :
#ifdef	DEBUG
			SC_log(LOG_DEBUG, "__SCDynamicStoreCopyValue() failed: %s", SCErrorString(sc_status));
#endif	/* DEBUG */
			goto done;
	}

	/*
	 * Save the new key.
	 */
	sc_status = __SCDynamicStoreSetValue(store, key, value, TRUE);

	/* push changes */
	__SCDynamicStorePush();

    done:

	return sc_status;
}


__private_extern__
kern_return_t
_configadd(mach_port_t 			server,
	   xmlData_t			keyRef,		/* raw XML bytes */
	   mach_msg_type_number_t	keyLen,
	   xmlData_t			dataRef,	/* raw XML bytes */
	   mach_msg_type_number_t	dataLen,
	   int				*newInstance,
	   int				*sc_status)
{
	CFStringRef		key		= NULL;		/* key  (un-serialized) */
	CFDataRef		data		= NULL;		/* data (un-serialized) */
	serverSessionRef	mySession;
	int			status;

	*newInstance = 0;
	*sc_status = kSCStatusOK;

	/* un-serialize the key */
	if (!_SCUnserializeString(&key, NULL, keyRef, keyLen)) {
		*sc_status = kSCStatusFailed;
	}

	/* un-serialize the data */
	if (!_SCUnserializeData(&data, dataRef, dataLen)) {
		*sc_status = kSCStatusFailed;
	}

	if (*sc_status != kSCStatusOK) {
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

		SC_trace("!add %s : %5u : %@",
			 storePrivate->useSessionKeys ? "t " : "  ",
			 storePrivate->server,
			 key);
#endif	// DEBUG
		*sc_status = status;
		goto done;
	}

	*sc_status = __SCDynamicStoreAddValue(mySession->store, key, data);

    done :

	if (key != NULL)	CFRelease(key);
	if (data != NULL)	CFRelease(data);

	return KERN_SUCCESS;
}


__private_extern__
kern_return_t
_configadd_s(mach_port_t 		server,
	     xmlData_t			keyRef,		/* raw XML bytes */
	     mach_msg_type_number_t	keyLen,
	     xmlData_t			dataRef,	/* raw XML bytes */
	     mach_msg_type_number_t	dataLen,
	     int			*newInstance,
	     int			*sc_status)
{
	CFDataRef			data		= NULL;		/* data (un-serialized) */
	CFStringRef			key		= NULL;		/* key  (un-serialized) */
	serverSessionRef		mySession;
	int				status;
	SCDynamicStorePrivateRef	storePrivate;
	Boolean				useSessionKeys;

	*newInstance = 0;
	*sc_status = kSCStatusOK;

	/* un-serialize the key */
	if (!_SCUnserializeString(&key, NULL, keyRef, keyLen)) {
		*sc_status = kSCStatusFailed;
	}

	/* un-serialize the data */
	if (!_SCUnserializeData(&data, dataRef, dataLen)) {
		*sc_status = kSCStatusFailed;
	}

	if (*sc_status != kSCStatusOK) {
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
		storePrivate = (SCDynamicStorePrivateRef)mySession->store;

		SC_trace("!add t  : %5u : %@",
			 storePrivate->server,
			 key);
#endif	// DEBUG
		*sc_status = status;
		goto done;
	}

	// force "useSessionKeys"
	storePrivate = (SCDynamicStorePrivateRef)mySession->store;
	useSessionKeys = storePrivate->useSessionKeys;
	storePrivate->useSessionKeys = TRUE;

	*sc_status = __SCDynamicStoreAddValue(mySession->store, key, data);

	// restore "useSessionKeys"
	storePrivate->useSessionKeys = useSessionKeys;

    done :

	if (key != NULL)	CFRelease(key);
	if (data != NULL)	CFRelease(data);

	return KERN_SUCCESS;
}
