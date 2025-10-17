/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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

#ifndef _S_SESSION_H
#define _S_SESSION_H

#include <sys/cdefs.h>
#include <os/availability.h>

#define DISPATCH_MACH_SPI 1
#include <dispatch/private.h>

/*
 * SCDynamicStore read no-fault entitlement
 *
 *   Key   : "com.apple.SystemConfiguration.SCDynamicStore-read-no-fault"
 *   Value : Boolean
 *             TRUE == do not issue a fault (or simulated crash) if a read
 *		       operation is denied due to a missing entitlement
 */
#define kSCReadNoFaultEntitlementName	CFSTR("com.apple.SystemConfiguration.SCDynamicStore-read-no-fault")

/*
 * SCDynamicStore write access entitlement
 *
 *   Key   : "com.apple.SystemConfiguration.SCDynamicStore-write-access"
 *   Value : Boolean
 *             TRUE == allow SCDynamicStore write access for this process
 *
 *           Dictionary
 *             Key   : "keys"
 *             Value : <array> of CFString with write access allowed for
 *                     each SCDynamicStore key matching the string(s)
 *
 *             Key   : "patterns"
 *             Value : <array> of CFString with write access allowed for
 *                     each SCDynamicStore key matching the regex pattern(s)
 */
#define	kSCWriteEntitlementName	CFSTR("com.apple.SystemConfiguration.SCDynamicStore-write-access")

/*
 * SCDynamicStore write no-fault entitlement
 *
 *   Key   : "com.apple.SystemConfiguration.SCDynamicStore-write-no-fault"
 *   Value : Boolean
 *             TRUE == do not issue a fault (or simulated crash) if a write
 *		       operation is denied due to a missing entitlement
 */
#define kSCWriteNoFaultEntitlementName	CFSTR("com.apple.SystemConfiguration.SCDynamicStore-write-no-fault")

/* Per client server state */
typedef struct {

	// base CFType information
	CFRuntimeBase           cfBase;

	/* mach port used as the key to this session */
	mach_port_t		key;

	/* mach channel associated with this session */
	dispatch_mach_t		serverChannel;

	/* data associated with this "open" session */
	CFMutableArrayRef	changedKeys;
	CFStringRef		name;
	CFMutableArrayRef	sessionKeys;
	SCDynamicStoreRef	store;

	/* credentials associated with this "open" session */
	uid_t			callerEUID;

	/* Mach security audit trailer for evaluating credentials */
	audit_token_t		auditToken;

	/*
	 * entitlements associated with this "open" session
	 *
	 * Note: the dictionary key is the entitlement name.  A
	 *       value of kCFNull indicates that the entitlement
	 *       does not exist for the session.
	 */
	CFMutableDictionaryRef	entitlements;

	/*
	 * isBackgroundAssetExtension
	 * - NULL means we haven't checked yet
	 * - kCFBooleanTrue/kCFBooleanFalse otherwise
	 * - not retained
	 */
	CFBooleanRef		isBackgroundAssetExtension;

	/*
	 * isPlatformBinary
	 * - NULL means we haven't checked yet
	 * - kCFBooleanTrue/kCFBooleanFalse otherwise
	 * - not retained
	 */
	CFBooleanRef		isPlatformBinary;

	/*
	 * isSystemProcess
	 * - NULL means we haven't checked yet
	 * - kCFBooleanTrue/kCFBooleanFalse otherwise
	 * - not retained
	 */
	CFBooleanRef		isSystemProcess;
} serverSession, *serverSessionRef;

__BEGIN_DECLS

serverSessionRef	addClient	(mach_port_t	server,
					 audit_token_t	audit_token);

serverSessionRef	addServer	(mach_port_t	server);

serverSessionRef	getSession	(mach_port_t	server);

serverSessionRef	getSessionNum	(CFNumberRef	serverKey);

serverSessionRef	getSessionStr	(CFStringRef	serverKey);

void			cleanupSession	(serverSessionRef	session);

void			closeSession	(serverSessionRef	session);

void			listSessions	(FILE		*f);

Boolean			hasRootAccess	(serverSessionRef	session);

int			checkReadAccess	(serverSessionRef	session,
					 CFStringRef		key,
					 CFDictionaryRef	controls);

int			checkWriteAccess(serverSessionRef	session,
					 CFStringRef		key);

__END_DECLS

#endif	/* !_S_SESSION_H */
