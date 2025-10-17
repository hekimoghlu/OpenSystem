/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "APIWebPushDaemonConnection.h"

#include "MessageSenderInlines.h"
#include "PushClientConnectionMessages.h"
#include "WebPushMessage.h"
#include <WebCore/ExceptionData.h>
#include <WebCore/NotificationData.h>
#include <WebCore/PushPermissionState.h>
#include <WebCore/PushSubscriptionData.h>

namespace API {

WebPushDaemonConnection::WebPushDaemonConnection(const WTF::String& machServiceName, WebKit::WebPushD::WebPushDaemonConnectionConfiguration&& configuration)
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    : m_connection(WebKit::WebPushD::Connection::create(machServiceName.utf8(), WTFMove(configuration)))
#endif
{
}

void WebPushDaemonConnection::getPushPermissionState(const WTF::URL& scopeURL, CompletionHandler<void(WebCore::PushPermissionState)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::GetPushPermissionState(SecurityOriginData::fromURL(scopeURL)), WTFMove(completionHandler));
#else
    completionHandler(WebCore::PushPermissionState::Denied);
#endif
}

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
Ref<WebKit::WebPushD::Connection> WebPushDaemonConnection::protectedConnection() const
{
    return m_connection;
}
#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)

void WebPushDaemonConnection::requestPushPermission(const WTF::URL& scopeURL, CompletionHandler<void(bool)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::RequestPushPermission(SecurityOriginData::fromURL(scopeURL)), WTFMove(completionHandler));
#else
    completionHandler(false);
#endif
}

void WebPushDaemonConnection::setAppBadge(const WTF::URL& scopeURL, std::optional<uint64_t> badge)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithoutUsingIPCConnection(Messages::PushClientConnection::SetAppBadge(SecurityOriginData::fromURL(scopeURL), badge));
#endif
}

void WebPushDaemonConnection::subscribeToPushService(const WTF::URL& scopeURL, const Vector<uint8_t>& applicationServerKey, CompletionHandler<void(const Expected<WebCore::PushSubscriptionData, WebCore::ExceptionData>&)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::SubscribeToPushService(scopeURL, applicationServerKey), WTFMove(completionHandler));
#else
    completionHandler(makeUnexpected(WebCore::ExceptionData { WebCore::ExceptionCode::UnknownError, "Cannot subscribe to push service"_s }));
#endif
}

void WebPushDaemonConnection::unsubscribeFromPushService(const WTF::URL& scopeURL, CompletionHandler<void(const Expected<bool, WebCore::ExceptionData>&)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::UnsubscribeFromPushService(scopeURL, std::nullopt), WTFMove(completionHandler));
#else
    completionHandler(false);
#endif
}

void WebPushDaemonConnection::getPushSubscription(const WTF::URL& scopeURL, CompletionHandler<void(const Expected<std::optional<WebCore::PushSubscriptionData>, WebCore::ExceptionData>&)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::GetPushSubscription(WTFMove(scopeURL)), WTFMove(completionHandler));
#else
    completionHandler(std::optional<WebCore::PushSubscriptionData> { });
#endif
}

void WebPushDaemonConnection::getNextPendingPushMessage(CompletionHandler<void(const std::optional<WebKit::WebPushMessage>&)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::GetPendingPushMessage(), WTFMove(completionHandler));
#else
    completionHandler(std::nullopt);
#endif
}

void WebPushDaemonConnection::showNotification(const WebCore::NotificationData& notificationData, CompletionHandler<void()>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::ShowNotification { notificationData, { } }, WTFMove(completionHandler));
#else
    completionHandler();
#endif
}

void WebPushDaemonConnection::getNotifications(const WTF::URL& scopeURL, const WTF::String& tag, CompletionHandler<void(const Expected<Vector<WebCore::NotificationData>, WebCore::ExceptionData>&)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::GetNotifications { scopeURL, tag }, WTFMove(completionHandler));
#else
    completionHandler({ });
#endif
}

void WebPushDaemonConnection::cancelNotification(const WTF::URL& scopeURL, const WTF::UUID& uuid)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    protectedConnection()->sendWithoutUsingIPCConnection(Messages::PushClientConnection::CancelNotification(SecurityOriginData::fromURL(scopeURL), uuid));
#endif
}

} // namespace API

