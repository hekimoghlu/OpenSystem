/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#ifndef _EAP8021X_EAPKEYCHAINUTIL_H
#define _EAP8021X_EAPKEYCHAINUTIL_H

/*
 * EAPKeychainUtil.h
 * - routines to deal with keychain passwords
 */

/* 
 * Modification History
 *
 * May 10, 2006		Dieter Siegmund (dieter@apple)
 * - created
 * December 4, 2009	Dieter Siegmund (dieter@apple)
 * - updated to use CFDictionary for passing values
 */

#include <TargetConditionals.h>
#include <os/availability.h>
#include <CoreFoundation/CoreFoundation.h>
#if ! TARGET_OS_IPHONE
#include <CoreServices/../Frameworks/CarbonCore.framework/Headers/MacErrors.h>
#endif /* ! TARGET_OS_IPHONE */

typedef CFTypeRef EAPSecAccessRef;

/*
 * Keys relevant to the values dict.
 */
#if ! TARGET_OS_IPHONE

/*
 * Note:
 *   These two properties are only consulted when doing a Create
 */
const CFStringRef kEAPSecKeychainPropTrustedApplications; /* CFArray[SecTrustedApplication] */
const CFStringRef kEAPSecKeychainPropAllowRootAccess;	  /* CFBoolean */

/*
 * In the Set APIs, specifying a value of kCFNull will remove any of the
 * following properties.
 */
const CFStringRef kEAPSecKeychainPropLabel;	/* CFData */
const CFStringRef kEAPSecKeychainPropDescription; /* CFData */
const CFStringRef kEAPSecKeychainPropAccount; 	/* CFData */
const CFStringRef kEAPSecKeychainPropPassword; 	/* CFData */

#endif /* ! TARGET_OS_IPHONE */

OSStatus
EAPSecKeychainPasswordItemCreateWithAccess(SecKeychainRef keychain,
					   EAPSecAccessRef access,
					   CFStringRef unique_id_str,
					   CFDataRef label,
					   CFDataRef description,
					   CFDataRef user,
					   CFDataRef password);
OSStatus
EAPSecKeychainPasswordItemCreateUniqueWithAccess(SecKeychainRef keychain,
						 EAPSecAccessRef access,
						 CFDataRef label,
						 CFDataRef description,
						 CFDataRef user,
						 CFDataRef password,
						 CFStringRef * unique_id_str);
OSStatus
EAPSecKeychainPasswordItemCreate(SecKeychainRef keychain,
				 CFStringRef unique_id_str,
				 CFDictionaryRef values) API_AVAILABLE(macos(10.10)) API_UNAVAILABLE(ios, watchos, tvos);
OSStatus
EAPSecKeychainPasswordItemCreateUnique(SecKeychainRef keychain,
				       CFDictionaryRef values,
				       CFStringRef * req_unique_id) API_AVAILABLE(macos(10.10)) API_UNAVAILABLE(ios, watchos, tvos);
OSStatus
EAPSecKeychainPasswordItemSet(SecKeychainRef keychain,
			      CFStringRef unique_id_str,
			      CFDataRef password);
OSStatus
EAPSecKeychainPasswordItemSet2(SecKeychainRef keychain,
			       CFStringRef unique_id_str,
			       CFDictionaryRef values) API_AVAILABLE(macos(10.10)) API_UNAVAILABLE(ios, watchos, tvos);
OSStatus
EAPSecKeychainPasswordItemCopy(SecKeychainRef keychain,
			       CFStringRef unique_id_str,
			       CFDataRef * ret_password);
OSStatus
EAPSecKeychainPasswordItemCopy2(SecKeychainRef keychain,
				CFStringRef unique_id_str,
				CFArrayRef keys,
				CFDictionaryRef * ret_values) API_AVAILABLE(macos(10.10)) API_UNAVAILABLE(ios, watchos, tvos);
OSStatus
EAPSecKeychainPasswordItemRemove(SecKeychainRef keychain,
				 CFStringRef unique_id_str);
#endif /* _EAP8021X_EAPKEYCHAINUTIL_H */

