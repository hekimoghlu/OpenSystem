/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#pragma once

#include "NotificationManagerMessageHandler.h"
#include <WebCore/PageIdentifier.h>
#include <pal/SessionID.h>
#include <wtf/HashMap.h>

namespace WebKit {

class WebsiteDataStore;

struct SharedPreferencesForWebProcess;

class ServiceWorkerNotificationHandler final : public NotificationManagerMessageHandler {
public:
    static ServiceWorkerNotificationHandler& singleton();

    // Do nothing since this is a singleton.
    void ref() const final { }
    void deref() const final { }

    void showNotification(IPC::Connection&, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&) final;

    void cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID) final;
    void clearNotifications(const Vector<WTF::UUID>& notificationIDs) final;
    void didDestroyNotification(const WTF::UUID& notificationID) final;
    void pageWasNotifiedOfNotificationPermission() final { }
    void requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&) final;
    void setAppBadge(const WebCore::SecurityOriginData&, std::optional<uint64_t> badge) final { }
    void getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;
    void getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(const IPC::Connection&) const final;
    bool handlesNotification(WTF::UUID value) const { return m_notificationToSessionMap.contains(value); }

private:
    explicit ServiceWorkerNotificationHandler() = default;

    WebsiteDataStore* dataStoreForNotificationID(const WTF::UUID&);

    HashMap<WTF::UUID, PAL::SessionID> m_notificationToSessionMap;
};

} // namespace WebKit
