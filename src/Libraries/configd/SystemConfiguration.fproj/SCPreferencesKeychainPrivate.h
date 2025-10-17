/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#ifndef	_SCPREFERENCESKEYCHAINPRIVATE_H
#define	_SCPREFERENCESKEYCHAINPRIVATE_H

/*
 * SCPreferencesKeychainPrivate.h
 * - routines to deal with keychain passwords
 */

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPreferences.h>
#include <Security/Security.h>

#pragma mark -
#pragma mark Keychain helper APIs

#define	kSCKeychainOptionsAllowRoot		CFSTR("AllowRoot")		// CFBoolean, allow uid==0 applications
#define kSCKeychainOptionsAllowedExecutables	CFSTR("AllowedExecutables")	// CFArray[CFURL]

__BEGIN_DECLS

SecKeychainRef
_SCSecKeychainCopySystemKeychain		(void)					API_AVAILABLE(macos(10.5), ios(2.0));

CFDataRef
_SCSecKeychainPasswordItemCopy			(SecKeychainRef		keychain,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCSecKeychainPasswordItemExists		(SecKeychainRef		keychain,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCSecKeychainPasswordItemRemove		(SecKeychainRef		keychain,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCSecKeychainPasswordItemSet			(SecKeychainRef		keychain,
						 CFStringRef		unique_id,
						 CFStringRef		label,
						 CFStringRef		description,
						 CFStringRef		account,
						 CFDataRef		password,
						 CFDictionaryRef	options)	API_AVAILABLE(macos(10.5), ios(2.0));


#pragma mark -
#pragma mark "System" Keychain APIs (w/SCPreferences)


CFDataRef
_SCPreferencesSystemKeychainPasswordItemCopy	(SCPreferencesRef	prefs,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCPreferencesSystemKeychainPasswordItemExists	(SCPreferencesRef	prefs,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCPreferencesSystemKeychainPasswordItemRemove	(SCPreferencesRef	prefs,
						 CFStringRef		unique_id)	API_AVAILABLE(macos(10.5), ios(2.0));

Boolean
_SCPreferencesSystemKeychainPasswordItemSet	(SCPreferencesRef	prefs,
						 CFStringRef		unique_id,
						 CFStringRef		label,
						 CFStringRef		description,
						 CFStringRef		account,
						 CFDataRef		password,
						 CFDictionaryRef	options)	API_AVAILABLE(macos(10.5), ios(2.0));

__END_DECLS

#endif	// _SCPREFERENCESKEYCHAINPRIVATE_H
