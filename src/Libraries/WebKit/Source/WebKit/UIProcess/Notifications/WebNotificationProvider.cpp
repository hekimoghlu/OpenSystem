/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "WebNotificationProvider.h"

#include "APIArray.h"
#include "APIDictionary.h"
#include "APINumber.h"
#include "APISecurityOrigin.h"
#include "WKAPICast.h"
#include "WebNotification.h"
#include "WebNotificationManagerProxy.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebNotificationProvider);

WebNotificationProvider::WebNotificationProvider(const WKNotificationProviderBase* provider)
{
    initialize(provider);
}

bool WebNotificationProvider::show(WebPageProxy* page, WebNotification& notification, RefPtr<WebCore::NotificationResources>&&)
{
    if (!m_client.show)
        return false;

    m_client.show(toAPI(page), toAPI(&notification), m_client.base.clientInfo);
    return true;
}

void WebNotificationProvider::cancel(WebNotification& notification)
{
    if (!m_client.cancel)
        return;

    m_client.cancel(toAPI(&notification), m_client.base.clientInfo);
}

void WebNotificationProvider::didDestroyNotification(WebNotification& notification)
{
    if (!m_client.didDestroyNotification)
        return;

    m_client.didDestroyNotification(toAPI(&notification), m_client.base.clientInfo);
}

void WebNotificationProvider::clearNotifications(const Vector<WebNotificationIdentifier>& notificationIDs)
{
    if (!m_client.clearNotifications)
        return;

    auto arrayIDs = notificationIDs.map([](auto& notificationID) -> RefPtr<API::Object> {
        return API::UInt64::create(notificationID.toUInt64());
    });
    m_client.clearNotifications(toAPI(API::Array::create(WTFMove(arrayIDs)).ptr()), m_client.base.clientInfo);
}

void WebNotificationProvider::addNotificationManager(WebNotificationManagerProxy& manager)
{
    if (!m_client.addNotificationManager)
        return;

    m_client.addNotificationManager(toAPI(&manager), m_client.base.clientInfo);
}

void WebNotificationProvider::removeNotificationManager(WebNotificationManagerProxy& manager)
{
    if (!m_client.removeNotificationManager)
        return;

    m_client.removeNotificationManager(toAPI(&manager), m_client.base.clientInfo);
}

HashMap<WTF::String, bool> WebNotificationProvider::notificationPermissions()
{
    HashMap<WTF::String, bool> permissions;
    if (!m_client.notificationPermissions)
        return permissions;

    RefPtr<API::Dictionary> knownPermissions = adoptRef(toImpl(m_client.notificationPermissions(m_client.base.clientInfo)));
    if (!knownPermissions)
        return permissions;

    Ref<API::Array> knownOrigins = knownPermissions->keys();
    for (size_t i = 0; i < knownOrigins->size(); ++i) {
        RefPtr origin = knownOrigins->at<API::String>(i);
        permissions.set(origin->string(), knownPermissions->get<API::Boolean>(origin->string())->value());
    }
    return permissions;
}

} // namespace WebKit
