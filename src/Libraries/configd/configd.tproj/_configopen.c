/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#include "configd_server.h"
#include "session.h"

#include <bsm/libbsm.h>
#include <sys/types.h>
#include <unistd.h>

__private_extern__
int
__SCDynamicStoreOpen(SCDynamicStoreRef *store, CFStringRef name)
{
	/*
	 * allocate and initialize a new session
	 */
	*store = (SCDynamicStoreRef)__SCDynamicStoreCreatePrivate(NULL, name, NULL, NULL);

	return kSCStatusOK;
}


__private_extern__
kern_return_t
_configopen(mach_port_t			server,
	    xmlData_t			nameRef,		/* raw XML bytes */
	    mach_msg_type_number_t	nameLen,
	    xmlData_t			optionsRef,		/* raw XML bytes */
	    mach_msg_type_number_t	optionsLen,
	    mach_port_t			*newServer,
	    int				*sc_status,
	    audit_token_t		audit_token)
{
	serverSessionRef		mySession;
	CFStringRef			name		= NULL;	/* name (un-serialized) */
	CFDictionaryRef			options		= NULL;	/* options (un-serialized) */
	SCDynamicStorePrivateRef	storePrivate;
	CFBooleanRef			useSessionKeys	= NULL;

	*newServer = MACH_PORT_NULL;
	*sc_status = kSCStatusOK;

	/* un-serialize the name */
	if (!_SCUnserializeString(&name, NULL, nameRef, nameLen)) {
		*sc_status = kSCStatusFailed;
	}

	if ((optionsRef != NULL) && (optionsLen > 0)) {
		/* un-serialize the [session] options */
		if (!_SCUnserialize((CFPropertyListRef *)&options, NULL, optionsRef, optionsLen)) {
			*sc_status = kSCStatusFailed;
		}
	}

	if (*sc_status != kSCStatusOK) {
		goto done;
	}

	if (!isA_CFString(name)) {
		*sc_status = kSCStatusInvalidArgument;
		goto done;
	}

	if (options != NULL) {
		if (!isA_CFDictionary(options)) {
			*sc_status = kSCStatusInvalidArgument;
			goto done;
		}

		/*
		 * [pre-]process any provided options
		 */
		useSessionKeys = CFDictionaryGetValue(options, kSCDynamicStoreUseSessionKeys);
		if (useSessionKeys != NULL) {
			if (!isA_CFBoolean(useSessionKeys)) {
				*sc_status = kSCStatusInvalidArgument;
				goto done;
			}
		}
	}

	/*
	 * establish the new session
	 */
	mySession = addClient(server, audit_token);
	if (mySession == NULL) {
		SC_log(LOG_NOTICE, "session is already open");
		*sc_status = kSCStatusFailed;	/* you can't re-open an "open" session */
		goto done;
	}

	*newServer = mySession->key;
	__MACH_PORT_DEBUG(TRUE, "*** _configopen (after addClient)", *newServer);

	SC_trace("open    : %5u : %@",
		 *newServer,
		 name);

	*sc_status = __SCDynamicStoreOpen(&mySession->store, name);
	storePrivate = (SCDynamicStorePrivateRef)mySession->store;

	/*
	 * Make the server port accessible to the framework routines.
	 * ... and be sure to clear before calling CFRelease(store)
	 */
	storePrivate->server = *newServer;

	/*
	 * Process any provided [session] options
	 */
	if (useSessionKeys != NULL) {
		storePrivate->useSessionKeys = CFBooleanGetValue(useSessionKeys);
	}

	/*
	 * Save the name of the calling application / plug-in with the session data.
	 */
	mySession->name = name;

	/*
	 * Note: at this time we should be holding ONE send right and
	 *       ONE receive right to the server.  The send right is
	 *       moved to the caller.
	 */

    done :

	if (options != NULL)	CFRelease(options);
	return KERN_SUCCESS;
}
