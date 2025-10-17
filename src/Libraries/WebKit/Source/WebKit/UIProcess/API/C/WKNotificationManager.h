/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#ifndef WKNotificationManager_h
#define WKNotificationManager_h

#include <WebKit/WKBase.h>
#include <WebKit/WKNotificationProvider.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKNotificationManagerGetTypeID();
WK_EXPORT void WKNotificationManagerSetProvider(WKNotificationManagerRef managerRef, const WKNotificationProviderBase* wkProvider);

WK_EXPORT void WKNotificationManagerProviderDidShowNotification(WKNotificationManagerRef managerRef, uint64_t notificationID);
WK_EXPORT void WKNotificationManagerProviderDidClickNotification(WKNotificationManagerRef managerRef, uint64_t notificationID);
WK_EXPORT void WKNotificationManagerProviderDidClickNotification_b(WKNotificationManagerRef managerRef, WKDataRef notificationID);
WK_EXPORT void WKNotificationManagerProviderDidCloseNotifications(WKNotificationManagerRef managerRef, WKArrayRef notificationIDs);
WK_EXPORT void WKNotificationManagerProviderDidUpdateNotificationPolicy(WKNotificationManagerRef managerRef, WKSecurityOriginRef origin, bool allowed);
WK_EXPORT void WKNotificationManagerProviderDidRemoveNotificationPolicies(WKNotificationManagerRef managerRef, WKArrayRef origins);

WK_EXPORT WKNotificationManagerRef WKNotificationManagerGetSharedServiceWorkerNotificationManager();

#ifdef __cplusplus
}
#endif

#endif // WKNotificationManager_h
