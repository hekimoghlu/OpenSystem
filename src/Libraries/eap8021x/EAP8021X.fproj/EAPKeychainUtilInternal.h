/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#ifndef _EAP8021X_EAPKEYCHAINUTILINTERNAL_H
#define _EAP8021X_EAPKEYCHAINUTILINTERNAL_H

/*
 * EAPKeychainUtilInternal.h
 * - internal definitions for setting keychain items
 */

/* 
 * Modification History
 *
 * January 14, 2010	Dieter Siegmund (dieter@apple)
 * - created
 */

#include <TargetConditionals.h>

#if ! TARGET_OS_IPHONE

#include "symbol_scope.h"
#include <Security/SecAccess.h>

PRIVATE_EXTERN OSStatus
EAPSecKeychainItemSetAccessForTrustedApplications(SecKeychainItemRef item,
						  CFArrayRef trusted_apps);
#endif /* ! TARGET_OS_IPHONE */

#if TARGET_OS_IPHONE

OSStatus
EAPKeychainSetIdentityReference(CFStringRef unique_string, CFDataRef reference, Boolean update);

OSStatus
EAPKeychainCopyIdentityReference(CFStringRef unique_string, CFDataRef *reference);

OSStatus
EAPKeychainSetPasswordItem(CFStringRef unique_string, CFDataRef username, CFDataRef password, Boolean update);

OSStatus
EAPKeychainCopyPasswordItem(CFStringRef unique_string, CFDataRef *username_p, CFDataRef *password_p);

OSStatus
EAPKeychainRemovePasswordItem(CFStringRef unique_string);

#endif /* TARGET_OS_IPHONE */

#endif /* _EAP8021X_EAPKEYCHAINUTILINTERNAL_H */

