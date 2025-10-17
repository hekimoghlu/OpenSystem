/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#include "ServiceWorkerNotificationHandler.h"

#include "Logging.h"
#include "WebProcessProxy.h"
#include "WebsiteDataStore.h"
#include <WebCore/NotificationData.h>
#include <wtf/Scope.h>

namespace WebKit {

ServiceWorkerNotificationHandler& ServiceWorkerNotificationHandler::singleton()
{
    ASSERT(isMainRunLoop());
    static ServiceWorkerNotificationHandler& handler = *new ServiceWorkerNotificationHandler;
    return handler;
}

WebsiteDataStore* ServiceWorkerNotificationHandler::dataStoreForNotificationID(const WTF::UUID& notificationID)
{
    auto iterator = m_notificationToSessionMap.find(notificationID);
    if (iterator == m_notificationToSessionMap.end())
        return nullptr;

    return WebsiteDataStore::existingDataStoreForSessionID(iterator->value);
}

void ServiceWorkerNotificationHandler::showNotification(IPC::Connection& connection, const WebCore::NotificationData& data, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&& callback)
{
    RELEASE_LOG(Push, "ServiceWorkerNotificationHandler showNotification called");

    auto scope = makeScopeExit([&callback] { callback(); });

    auto* dataStore = WebsiteDataStore::existingDataStoreForSessionID(data.sourceSession);
    if (!dataStore)
        return;

    m_notificationToSessionMap.add(data.notificationID, data.sourceSession);
    dataStore->showPersistentNotification(&connection, data);
}

void ServiceWorkerNotificationHandler::cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID)
{
    if (auto* dataStore = dataStoreForNotificationID(notificationID))
        dataStore->cancelServiceWorkerNotification(notificationID);
}

void ServiceWorkerNotificationHandler::clearNotifications(const Vector<WTF::UUID>& notificationIDs)
{
    for (auto& notificationID : notificationIDs) {
        if (auto* dataStore = dataStoreForNotificationID(notificationID))
            dataStore->clearServiceWorkerNotification(notificationID);
    }
}

void ServiceWorkerNotificationHandler::didDestroyNotification(const WTF::UUID& notificationID)
{
    if (auto* dataStore = dataStoreForNotificationID(notificationID))
        dataStore->didDestroyServiceWorkerNotification(notificationID);
}

void ServiceWorkerNotificationHandler::requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void ServiceWorkerNotificationHandler::getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void ServiceWorkerNotificationHandler::getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

std::optional<SharedPreferencesForWebProcess> ServiceWorkerNotificationHandler::sharedPreferencesForWebProcess(const IPC::Connection& connection) const
{
    if (auto webProcessProxy = WebProcessProxy::processForConnection(connection))
        return webProcessProxy->sharedPreferencesForWebProcess();

    ASSERT_NOT_REACHED();
    return std::nullopt;
}

} // namespace WebKit
