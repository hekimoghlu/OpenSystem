/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "WKNotification.h"

#include "APISecurityOrigin.h"
#include "WKAPICast.h"
#include "WKData.h"
#include "WKString.h"
#include "WebNotification.h"
#include <WebCore/NotificationDirection.h>

using namespace WebKit;

WKTypeID WKNotificationGetTypeID()
{
    return toAPI(WebNotification::APIType);
}

WKStringRef WKNotificationCopyTitle(WKNotificationRef notification)
{
    return toCopiedAPI(toImpl(notification)->title());
}

WKStringRef WKNotificationCopyBody(WKNotificationRef notification)
{
    return toCopiedAPI(toImpl(notification)->body());
}

WKStringRef WKNotificationCopyIconURL(WKNotificationRef notification)
{
    return toCopiedAPI(toImpl(notification)->iconURL());
}

WKStringRef WKNotificationCopyTag(WKNotificationRef notification)
{
    return toCopiedAPI(toImpl(notification)->tag());
}

WKStringRef WKNotificationCopyLang(WKNotificationRef notification)
{
    return toCopiedAPI(toImpl(notification)->lang());
}

WKStringRef WKNotificationCopyDir(WKNotificationRef notification)
{
    switch (toImpl(notification)->dir()) {
    case WebCore::NotificationDirection::Auto:
        return WKStringCreateWithUTF8CString("auto");
    case WebCore::NotificationDirection::Ltr:
        return WKStringCreateWithUTF8CString("ltr");
    case WebCore::NotificationDirection::Rtl:
        return WKStringCreateWithUTF8CString("rtl");
    }

    RELEASE_ASSERT_NOT_REACHED();
}

WKSecurityOriginRef WKNotificationGetSecurityOrigin(WKNotificationRef notification)
{
    return toAPI(toImpl(notification)->origin());
}

uint64_t WKNotificationGetID(WKNotificationRef notification)
{
    return toImpl(notification)->identifier().toUInt64();
}

WKStringRef WKNotificationCopyDataStoreIdentifier(WKNotificationRef notification)
{
    auto identifier = toImpl(notification)->dataStoreIdentifier();
    return identifier ? toCopiedAPI(identifier->toString()) : nullptr;
}

WKDataRef WKNotificationCopyCoreIDForTesting(WKNotificationRef notification)
{
    auto identifier = toImpl(notification)->coreNotificationID();
    auto span = identifier.span();
    return WKDataCreate(span.data(), span.size());
}

bool WKNotificationGetIsPersistent(WKNotificationRef notification)
{
    return toImpl(notification)->isPersistentNotification();
}

WKNotificationAlert WKNotificationGetAlert(WKNotificationRef notification)
{
    auto silent = toImpl(notification)->data().silent;
    if (silent == std::nullopt)
        return kWKNotificationAlertDefault;
    return *silent ? kWKNotificationAlertSilent : kWKNotificationAlertEnabled;
}
