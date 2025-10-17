/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#ifndef WKNotification_h
#define WKNotification_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif


enum {
    kWKNotificationAlertDefault = 1 << 0,
    kWKNotificationAlertSilent = 1 << 1,
    kWKNotificationAlertEnabled = 1 << 2
};
typedef uint32_t WKNotificationAlert;

WK_EXPORT WKTypeID WKNotificationGetTypeID();

WK_EXPORT WKStringRef WKNotificationCopyTitle(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyBody(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyIconURL(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyTag(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyLang(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyDir(WKNotificationRef notification);
WK_EXPORT WKSecurityOriginRef WKNotificationGetSecurityOrigin(WKNotificationRef notification);
WK_EXPORT uint64_t WKNotificationGetID(WKNotificationRef notification);
WK_EXPORT WKStringRef WKNotificationCopyDataStoreIdentifier(WKNotificationRef notification);
WK_EXPORT WKDataRef WKNotificationCopyCoreIDForTesting(WKNotificationRef notification);
WK_EXPORT bool WKNotificationGetIsPersistent(WKNotificationRef notification);
WK_EXPORT WKNotificationAlert WKNotificationGetAlert(WKNotificationRef notification);

#ifdef __cplusplus
}
#endif

#endif // WKNotification_h
