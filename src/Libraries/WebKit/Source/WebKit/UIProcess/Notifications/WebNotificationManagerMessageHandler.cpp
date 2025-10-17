/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include "WebNotificationManagerMessageHandler.h"

#include "Logging.h"
#include "ServiceWorkerNotificationHandler.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/NotificationData.h>
#include <wtf/CompletionHandler.h>

namespace WebKit {

WebNotificationManagerMessageHandler::WebNotificationManagerMessageHandler(WebPageProxy& webPageProxy)
    : m_webPageProxy(webPageProxy)
{
}

void WebNotificationManagerMessageHandler::ref() const
{
    m_webPageProxy->ref();
}

void WebNotificationManagerMessageHandler::deref() const
{
    m_webPageProxy->deref();
}

Ref<WebPageProxy> WebNotificationManagerMessageHandler::protectedPage() const
{
    return m_webPageProxy.get();
}

void WebNotificationManagerMessageHandler::showNotification(IPC::Connection& connection, const WebCore::NotificationData& data, RefPtr<WebCore::NotificationResources>&& resources, CompletionHandler<void()>&& callback)
{
    RELEASE_LOG(Push, "WebNotificationManagerMessageHandler showNotification called");

    if (!data.serviceWorkerRegistrationURL.isEmpty()) {
        ServiceWorkerNotificationHandler::singleton().showNotification(connection, data, WTFMove(resources), WTFMove(callback));
        return;
    }
    protectedPage()->showNotification(connection, data, WTFMove(resources));
    callback();
}

void WebNotificationManagerMessageHandler::cancelNotification(WebCore::SecurityOriginData&& origin, const WTF::UUID& notificationID)
{
    Ref serviceWorkerNotificationHandler = ServiceWorkerNotificationHandler::singleton();
    if (serviceWorkerNotificationHandler->handlesNotification(notificationID)) {
        serviceWorkerNotificationHandler->cancelNotification(WTFMove(origin), notificationID);
        return;
    }
    protectedPage()->cancelNotification(notificationID);
}

void WebNotificationManagerMessageHandler::clearNotifications(const Vector<WTF::UUID>& notificationIDs)
{
    Ref serviceWorkerNotificationHandler = ServiceWorkerNotificationHandler::singleton();

    Vector<WTF::UUID> persistentNotifications;
    Vector<WTF::UUID> pageNotifications;
    persistentNotifications.reserveInitialCapacity(notificationIDs.size());
    pageNotifications.reserveInitialCapacity(notificationIDs.size());
    for (auto& notificationID : notificationIDs) {
        if (serviceWorkerNotificationHandler->handlesNotification(notificationID))
            persistentNotifications.append(notificationID);
        else
            pageNotifications.append(notificationID);
    }
    if (!persistentNotifications.isEmpty())
        serviceWorkerNotificationHandler->clearNotifications(persistentNotifications);
    if (!pageNotifications.isEmpty())
        protectedPage()->clearNotifications(pageNotifications);
}

void WebNotificationManagerMessageHandler::didDestroyNotification(const WTF::UUID& notificationID)
{
    Ref serviceWorkerNotificationHandler = ServiceWorkerNotificationHandler::singleton();
    if (serviceWorkerNotificationHandler->handlesNotification(notificationID)) {
        serviceWorkerNotificationHandler->didDestroyNotification(notificationID);
        return;
    }
    protectedPage()->didDestroyNotification(notificationID);
}

void WebNotificationManagerMessageHandler::pageWasNotifiedOfNotificationPermission()
{
    protectedPage()->pageWillLikelyUseNotifications();
}

void WebNotificationManagerMessageHandler::requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler({ });
}

void WebNotificationManagerMessageHandler::getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler({ });
}

void WebNotificationManagerMessageHandler::getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler({ });
}

std::optional<SharedPreferencesForWebProcess> WebNotificationManagerMessageHandler::sharedPreferencesForWebProcess(const IPC::Connection&) const
{
    return protectedPage()->protectedLegacyMainFrameProcess()->sharedPreferencesForWebProcess();
}

} // namespace WebKit
