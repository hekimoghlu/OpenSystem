/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#ifndef _EAP8021X_EAPOLCONTROLPREFS_H
#define _EAP8021X_EAPOLCONTROLPREFS_H

/*
 * EAPOLControlPrefs.h
 * - definitions for accessing EAPOL preferences and being notified
 *   when they change
 */

/* 
 * Modification History
 *
 * January 9, 2013	Dieter Siegmund (dieter@apple)
 * - created
 */
#include <CoreFoundation/CFRunLoop.h>
#include <SystemConfiguration/SCPreferences.h>

typedef void (*EAPOLControlPrefsCallBack)(SCPreferencesRef prefs);

SCPreferencesRef
EAPOLControlPrefsInit(CFRunLoopRef runloop, EAPOLControlPrefsCallBack callback);

void
EAPOLControlPrefsSynchronize(void);

uint32_t
EAPOLControlPrefsGetLogFlags(void);

Boolean
EAPOLControlPrefsSetLogFlags(uint32_t flags);

Boolean
EAPOLControlPrefsSetUseBoringSSL(bool use_boringssl);

Boolean
EAPOLControlPrefsGetUseBoringSSL(void);

Boolean
EAPOLControlPrefsSetRevocationCheck(bool enable);

Boolean
EAPOLControlPrefsGetRevocationCheck(void);

#endif /* _EAP8021X_EAPOLCONTROLPREFS_H */
