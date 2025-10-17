/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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

#include "APIObject.h"
#include "WebPushDaemonConnection.h"
#include <optional>

namespace WebCore {
enum class PushPermissionState : uint8_t;
struct ExceptionData;
struct NotificationData;
struct PushSubscriptionData;
}

namespace WebKit {
namespace WebPushD {
class Connection;
struct WebPushDaemonConnectionConfiguration;
}
struct WebPushMessage;
}

namespace API {

class WebPushDaemonConnection final : public ObjectImpl<Object::Type::WebPushDaemonConnection> {
public:
    WebPushDaemonConnection(const WTF::String& machServiceName, WebKit::WebPushD::WebPushDaemonConnectionConfiguration&&);

    void getPushPermissionState(const WTF::URL&, CompletionHandler<void(WebCore::PushPermissionState)>&&);
    void requestPushPermission(const WTF::URL&, CompletionHandler<void(bool)>&&);
    void setAppBadge(const WTF::URL&, std::optional<uint64_t>);
    void subscribeToPushService(const WTF::URL&, const Vector<uint8_t>& applicationServerKey, CompletionHandler<void(const Expected<WebCore::PushSubscriptionData, WebCore::ExceptionData>&)>&&);
    void unsubscribeFromPushService(const WTF::URL&, CompletionHandler<void(const Expected<bool, WebCore::ExceptionData>&)>&&);
    void getPushSubscription(const WTF::URL&, CompletionHandler<void(const Expected<std::optional<WebCore::PushSubscriptionData>, WebCore::ExceptionData>&)>&&);
    void getNextPendingPushMessage(CompletionHandler<void(const std::optional<WebKit::WebPushMessage>&)>&&);

    void showNotification(const WebCore::NotificationData&, CompletionHandler<void()>&&);
    void getNotifications(const WTF::URL&, const WTF::String& tag, CompletionHandler<void(const Expected<Vector<WebCore::NotificationData>, WebCore::ExceptionData>&)>&&);
    void cancelNotification(const WTF::URL&, const WTF::UUID&);

private:
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    Ref<WebKit::WebPushD::Connection> protectedConnection() const;

    Ref<WebKit::WebPushD::Connection> m_connection;
#endif
};

} // namespace API
