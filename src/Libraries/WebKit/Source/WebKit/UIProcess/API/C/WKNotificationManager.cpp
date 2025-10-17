/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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
#include "config.h"
#include "WKNotificationManager.h"

#include "APIArray.h"
#include "APIData.h"
#include "WKAPICast.h"
#include "WebNotification.h"
#include "WebNotificationManagerProxy.h"
#include "WebNotificationProvider.h"

using namespace WebKit;

WKTypeID WKNotificationManagerGetTypeID()
{
    return toAPI(WebNotificationManagerProxy::APIType);
}

void WKNotificationManagerSetProvider(WKNotificationManagerRef managerRef, const WKNotificationProviderBase* wkProvider)
{
    toImpl(managerRef)->setProvider(makeUnique<WebNotificationProvider>(wkProvider));
}

void WKNotificationManagerProviderDidShowNotification(WKNotificationManagerRef managerRef, uint64_t notificationID)
{
    toImpl(managerRef)->providerDidShowNotification(WebNotificationIdentifier { notificationID });
}

void WKNotificationManagerProviderDidClickNotification(WKNotificationManagerRef managerRef, uint64_t notificationID)
{
    toImpl(managerRef)->providerDidClickNotification(WebNotificationIdentifier { notificationID });
}

void WKNotificationManagerProviderDidClickNotification_b(WKNotificationManagerRef managerRef, WKDataRef identifier)
{
    auto span = toImpl(identifier)->span();
    if (span.size() != 16)
        return;

    toImpl(managerRef)->providerDidClickNotification(WTF::UUID { std::span<const uint8_t, 16> { span } });
}

void WKNotificationManagerProviderDidCloseNotifications(WKNotificationManagerRef managerRef, WKArrayRef notificationIDs)
{
    toImpl(managerRef)->providerDidCloseNotifications(toImpl(notificationIDs));
}

void WKNotificationManagerProviderDidUpdateNotificationPolicy(WKNotificationManagerRef managerRef, WKSecurityOriginRef origin, bool allowed)
{
    toImpl(managerRef)->providerDidUpdateNotificationPolicy(toImpl(origin), allowed);
}

void WKNotificationManagerProviderDidRemoveNotificationPolicies(WKNotificationManagerRef managerRef, WKArrayRef origins)
{
    toImpl(managerRef)->providerDidRemoveNotificationPolicies(toImpl(origins));
}

WKNotificationManagerRef WKNotificationManagerGetSharedServiceWorkerNotificationManager()
{
    return toAPI(&WebNotificationManagerProxy::sharedServiceWorkerManager());
}
