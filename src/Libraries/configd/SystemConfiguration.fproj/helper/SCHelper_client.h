/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#ifndef _SCHELPER_CLIENT_H
#define _SCHELPER_CLIENT_H

#include <sys/cdefs.h>
#include <os/availability.h>
#include <TargetConditionals.h>
#include <CoreFoundation/CoreFoundation.h>

#define kSCKeychainOptionsAccount		CFSTR("Account")		// CFString
#define kSCKeychainOptionsDescription		CFSTR("Description")		// CFString
#define kSCKeychainOptionsLabel			CFSTR("Label")			// CFString
#define kSCKeychainOptionsPassword		CFSTR("Password")		// CFData
#define kSCKeychainOptionsUniqueID		CFSTR("UniqueID")		// CFString

#define kSCHelperAuthAuthorization		CFSTR("Authorization")		// CFData[AuthorizationExternalForm]
#define kSCHelperAuthCallerInfo			CFSTR("CallerInfo")		// CFString

enum {
	// authorization
	SCHELPER_MSG_AUTH		= 1,

	// SCPreferences
	SCHELPER_MSG_PREFS_OPEN		= 100,
	SCHELPER_MSG_PREFS_ACCESS,
	SCHELPER_MSG_PREFS_LOCK,
	SCHELPER_MSG_PREFS_LOCKWAIT,
	SCHELPER_MSG_PREFS_COMMIT,
	SCHELPER_MSG_PREFS_APPLY,
	SCHELPER_MSG_PREFS_UNLOCK,
	SCHELPER_MSG_PREFS_CLOSE,
	SCHELPER_MSG_PREFS_SYNCHRONIZE,

	// SCNetworkConfiguration
	SCHELPER_MSG_INTERFACE_REFRESH	= 200,

#if	!TARGET_OS_IPHONE
	// "System" Keychain
	SCHELPER_MSG_KEYCHAIN_COPY	= 300,
	SCHELPER_MSG_KEYCHAIN_EXISTS,
	SCHELPER_MSG_KEYCHAIN_REMOVE,
	SCHELPER_MSG_KEYCHAIN_SET,
#endif	// !TARGET_OS_IPHONE

	// miscellaneous
	SCHELPER_MSG_EXIT		= 9999
};

__BEGIN_DECLS

Boolean	_SCHelperOpen	(CFDataRef		authorizationData,
			 mach_port_t		*helper_port);

Boolean	_SCHelperExec	(mach_port_t		helper_port,
			 uint32_t		msgID,
			 CFDataRef		data,
			 uint32_t		*status,
			 CFDataRef		*reply);

void	_SCHelperClose	(mach_port_t		*helper_port);

__END_DECLS

#endif	/* _SCHELPER_CLIENT_H */

