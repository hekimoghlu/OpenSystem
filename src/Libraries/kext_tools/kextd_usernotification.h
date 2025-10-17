/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#ifndef __KEXTD_USERNOTIFICATION_H__
#define __KEXTD_USERNOTIFICATION_H__

#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include "kextd_main.h"

#ifndef NO_CFUserNotification
ExitStatus startMonitoringConsoleUser(
    KextdArgs    * toolArgs);
void stopMonitoringConsoleUser(void);

#define INVALID_SIGNATURE_KEXT_ALERT       1
#define NO_LOAD_KEXT_ALERT                 2
#define REVOKED_SIG_KEXT_ALERT             3
#define EXCLUDED_KEXT_ALERT                4
//#define UNSIGNED_KEXT_ALERT                5

Boolean recordNonsecureKexts(CFArrayRef kextList);
Boolean recordNonSignedKextPath(CFStringRef theKextPath);
void    resetUserNotifications(Boolean dismissAlert);
void    sendNonsignedKextNotification(void);

void writeKextAlertPlist(CFDictionaryRef theDict, int theAlertType);
void writeKextLoadPlist(CFArrayRef theArray);
void sendRevokedCertAlert(CFDictionaryRef theDict);

void kextd_raise_notification(
    CFStringRef alertHeader,
    CFArrayRef  messageArray);

#endif /* ifndef NO_CFUserNotification */

#endif /* __KEXTD_USERNOTIFICATION_H__ */
