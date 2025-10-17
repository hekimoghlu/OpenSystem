/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#ifndef WKNotificationProvider_h
#define WKNotificationProvider_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKNotificationProviderShowCallback)(WKPageRef page, WKNotificationRef notification, const void* clientInfo);
typedef void (*WKNotificationProviderCancelCallback)(WKNotificationRef notification, const void* clientInfo);
typedef void (*WKNotificationProviderDidDestroyNotificationCallback)(WKNotificationRef notification, const void* clientInfo);
typedef void (*WKNotificationProviderAddNotificationManagerCallback)(WKNotificationManagerRef manager, const void* clientInfo);
typedef void (*WKNotificationProviderRemoveNotificationManagerCallback)(WKNotificationManagerRef manager, const void* clientInfo);
typedef WKDictionaryRef (*WKNotificationProviderNotificationPermissionsCallback)(const void* clientInfo);
typedef void (*WKNotificationProviderClearNotificationsCallback)(WKArrayRef notificationIDs, const void* clientInfo);

typedef struct WKNotificationProviderBase {
    int                                                                   version;
    const void*                                                           clientInfo;
} WKNotificationProviderBase;

typedef struct WKNotificationProviderV0 {
    WKNotificationProviderBase                                            base;

    // Version 0.
    WKNotificationProviderShowCallback                                    show;
    WKNotificationProviderCancelCallback                                  cancel;
    WKNotificationProviderDidDestroyNotificationCallback                  didDestroyNotification;
    WKNotificationProviderAddNotificationManagerCallback                  addNotificationManager;
    WKNotificationProviderRemoveNotificationManagerCallback               removeNotificationManager;
    WKNotificationProviderNotificationPermissionsCallback                 notificationPermissions;
    WKNotificationProviderClearNotificationsCallback                      clearNotifications;
} WKNotificationProviderV0;

#ifdef __cplusplus
}
#endif


#endif // WKNotificationProvider_h
