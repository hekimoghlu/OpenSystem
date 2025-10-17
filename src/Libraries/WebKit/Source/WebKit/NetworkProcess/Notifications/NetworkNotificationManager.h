/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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

#if ENABLE(WEB_PUSH_NOTIFICATIONS)

#include "NetworkProcess.h"
#include "NotificationManagerMessageHandler.h"
#include "SharedPreferencesForWebProcess.h"
#include "WebPushDaemonConnection.h"
#include "WebPushDaemonConnectionConfiguration.h"
#include "WebPushMessage.h"
#include <WebCore/ExceptionData.h>
#include <WebCore/NotificationDirection.h>
#include <WebCore/PushSubscriptionData.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class NotificationResources;
class SecurityOriginData;
}

namespace WebKit {

namespace WebPushD {
enum class MessageType : uint8_t;
}

class NetworkNotificationManager : public NotificationManagerMessageHandler, public RefCounted<NetworkNotificationManager> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkNotificationManager);
public:
    static Ref<NetworkNotificationManager> create(const String& webPushMachServiceName, WebPushD::WebPushDaemonConnectionConfiguration&&, NetworkProcess&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void setPushAndNotificationsEnabledForOrigin(const WebCore::SecurityOriginData&, bool, CompletionHandler<void()>&&);
    void getPendingPushMessage(CompletionHandler<void(const std::optional<WebPushMessage>&)>&&);
    void getPendingPushMessages(CompletionHandler<void(const Vector<WebPushMessage>&)>&&);

    void subscribeToPushService(URL&& scopeURL, Vector<uint8_t>&& applicationServerKey, CompletionHandler<void(Expected<WebCore::PushSubscriptionData, WebCore::ExceptionData>&&)>&&);
    void unsubscribeFromPushService(URL&& scopeURL, std::optional<WebCore::PushSubscriptionIdentifier>, CompletionHandler<void(Expected<bool, WebCore::ExceptionData>&&)>&&);
    void getPushSubscription(URL&& scopeURL, CompletionHandler<void(Expected<std::optional<WebCore::PushSubscriptionData>, WebCore::ExceptionData>&&)>&&);

    void requestPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&) final;
    void getPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;

    void incrementSilentPushCount(WebCore::SecurityOriginData&&, CompletionHandler<void(unsigned)>&&);
    void removeAllPushSubscriptions(CompletionHandler<void(unsigned)>&&);
    void removePushSubscriptionsForOrigin(WebCore::SecurityOriginData&&, CompletionHandler<void(unsigned)>&&);

    void showNotification(const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&);
    void getNotifications(const URL& registrationURL, const String& tag, CompletionHandler<void(Expected<Vector<WebCore::NotificationData>, WebCore::ExceptionData>&&)>&&);
    void clearNotifications(const Vector<WTF::UUID>& notificationIDs) final;

    void getAppBadgeForTesting(CompletionHandler<void(std::optional<uint64_t>)>&&);
    void setAppBadge(const WebCore::SecurityOriginData&, std::optional<uint64_t> badge) final;

private:
    NetworkNotificationManager(const String& webPushMachServiceName, WebPushD::WebPushDaemonConnectionConfiguration&&, NetworkProcess&);

    void showNotification(IPC::Connection&, const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>&&, CompletionHandler<void()>&&) final;
    void cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID) final;
    void didDestroyNotification(const WTF::UUID& notificationID) final;
    void pageWasNotifiedOfNotificationPermission() final { }
    void getPermissionStateSync(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&) final;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(const IPC::Connection&) const final;
    RefPtr<WebPushD::Connection> protectedConnection() const;

    RefPtr<WebPushD::Connection> m_connection;
    Ref<NetworkProcess> m_networkProcess;
};

} // namespace WebKit

#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)
