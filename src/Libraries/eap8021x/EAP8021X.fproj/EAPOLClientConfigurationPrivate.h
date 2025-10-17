/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#ifndef _EAP8021X_EAPOLCLIENTCONFIGURATIONPRIVATE_H
#define _EAP8021X_EAPOLCLIENTCONFIGURATIONPRIVATE_H

#include <EAP8021X/EAPOLClientConfiguration.h>
#include <CoreFoundation/CFRuntime.h>
#include <SystemConfiguration/SCPreferences.h>

/*
 * EAPOLClientConfigurationPrivate.h
 * - EAPOL client configuration private functions
 */

/* 
 * Modification History
 *
 * January 5, 2009	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/**
 ** EAPOLClientItemID
 **/
CFStringRef
EAPOLClientItemIDGetProfileID(EAPOLClientItemIDRef itemID);

CFDataRef
EAPOLClientItemIDGetWLANSSID(EAPOLClientItemIDRef itemID);

EAPOLClientProfileRef
EAPOLClientItemIDGetProfile(EAPOLClientItemIDRef itemID);

/*
 * Function: EAPOLClientItemIDCopyDictionary
 *           EAPOLClientItemIDCreateWithDictionary
 * Purpose:
 *   EAPOLClientItemIDCopyDictionary() creates an externalized form of the
 *   EAPOLClientItemIDRef that can be passed (after serialization) to another
 *   process.   The other process turns it back into an EAPOLClientItemIDRef
 *   by calling EAPOLClientItemIDCreateWithDictionary().
 */
CFDictionaryRef
EAPOLClientItemIDCopyDictionary(EAPOLClientItemIDRef itemID);

EAPOLClientItemIDRef
EAPOLClientItemIDCreateWithDictionary(EAPOLClientConfigurationRef cfg,
				      CFDictionaryRef dict);

#if ! TARGET_OS_IPHONE
OSStatus
EAPOLClientSetACLForIdentity(SecIdentityRef identity);
#endif /* ! TARGET_OS_IPHONE */

#endif /* _EAP8021X_EAPOLCLIENTCONFIGURATIONPRIVATE_H */
