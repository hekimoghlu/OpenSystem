/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#ifndef _EAP8021X_EAPOLCONTROLPRIVATE_H
#define _EAP8021X_EAPOLCONTROLPRIVATE_H

/* 
 * Modification History
 *
 * September 3, 2010	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * EAPOLControlPrivate.h
 * - EAPOLControl private definitions
 */

#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFDictionary.h>
#include <TargetConditionals.h>

#if ! TARGET_OS_IPHONE

extern const CFStringRef	kEAPOLAutoDetectSecondsSinceLastPacket; /* CFNumber */
extern const CFStringRef	kEAPOLAutoDetectAuthenticatorMACAddress; /* CFDataRef */

/*
 * Key: kEAPOLControlAutoDetectInformationNotifyKey
 * Purpose:
 *   SCDynamicStore notify key posted by the 802.1X auto-detection code
 *   whenever the auto-detect information has changed.
 */
extern const CFStringRef	kEAPOLControlAutoDetectInformationNotifyKey;

/*
 * Function: EAPOLControlCopyAutoDetectInformation
 *
 * Purpose:
 *   Returns a dictionary of (key, value) pairs.  The key is the interface
 *   name, the value is the number of seconds since having received the last
 *   802.1X packet.
 *
 * Returns:
 *   0 if successful, and *info_p contains a non-NULL CFDictionaryRef that
 *   must be released
 *   non-zero errno value otherwise, and *info_p is set to NULL.
 */
int
EAPOLControlCopyAutoDetectInformation(CFDictionaryRef * info_p);

extern const CFStringRef
kEAPOLControlStartOptionManagerName; /* CFStringRef */

extern const CFStringRef
kEAPOLControlStartOptionAuthenticationInfo; /* CFDictionary */

/*
 * Function: EAPOLControlStartWithOptions
 *
 * Purpose:
 *    Start an authentication session with the provided options, which may
 *    be NULL.
 */
int
EAPOLControlStartWithOptions(const char * if_name,
			     EAPOLClientItemIDRef itemID,
			     CFDictionaryRef options);

/*
 * Function: EAPOLControlCopyItemIDForAuthenticator
 * Purpose:
 *   Return the binding for the current user for the specified
 *   Authenticator.
 */
EAPOLClientItemIDRef
EAPOLControlCopyItemIDForAuthenticator(CFDataRef authenticator);

/*
 * Function: EAPOLControlSetItemIDForAuthenticator
 * Purpose:
 *   Set the binding for the current user for the specified
 *   Authenticator.
 *  
 *   Supplying an 'itemID' with value NULL clears the binding.
 */
void
EAPOLControlSetItemIDForAuthenticator(CFDataRef authenticator,
				      EAPOLClientItemIDRef itemID);

/*
 * Const: kEAPOLControlUserSettingsNotifyKey
 * Purpose:
 *   Notification key used with BSD notify(3) to let know whether user
 *   EAPOLControl settings have been modified.
 */
extern const char * kEAPOLControlUserSettingsNotifyKey;

#endif /* ! TARGET_OS_IPHONE */

#endif /* _EAP8021X_EAPOLCONTROLPRIVATE_H */
