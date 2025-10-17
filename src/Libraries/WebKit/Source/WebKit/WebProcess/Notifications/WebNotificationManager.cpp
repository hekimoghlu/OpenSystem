/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include "WebNotificationManager.h"

#include "Logging.h"
#include "WebPage.h"
#include "WebProcess.h"
#include "WebProcessCreationParameters.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(NOTIFICATIONS)
#include "NetworkProcessConnection.h"
#include "NotificationManagerMessageHandlerMessages.h"
#include "ServiceWorkerNotificationHandler.h"
#include "WebNotification.h"
#include "WebNotificationManagerMessages.h"
#include "WebPageProxyMessages.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/Document.h>
#include <WebCore/Notification.h>
#include <WebCore/NotificationData.h>
#include <WebCore/Page.h>
#include <WebCore/PushPermissionState.h>
#include <WebCore/SWContextManager.h>
#include <WebCore/ScriptExecutionContext.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/Settings.h>
#include <WebCore/UserGestureIndicator.h>
#endif

namespace WebKit {
using namespace WebCore;

#if ENABLE(NOTIFICATIONS)
static bool sendMessage(WebPage* page, const Function<bool(IPC::Connection&, uint64_t)>& sendMessage)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    if (DeprecatedGlobalSettings::builtInNotificationsEnabled()) {
        Ref networkProcessConnection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
        return sendMessage(networkProcessConnection, WebProcess::singleton().sessionID().toUInt64());
    }
#endif

    std::optional<WebCore::PageIdentifier> pageIdentifier;
    if (page)
        pageIdentifier = page->identifier();
    else if (auto* connection = SWContextManager::singleton().connection()) {
        // Pageless notification messages are, by default, on behalf of a service worker.
        // So use the service worker connection's page identifier.
        pageIdentifier = connection->pageIdentifier();
    }

    ASSERT(pageIdentifier);
    Ref parentConnection = *WebProcess::singleton().parentProcessConnection();
    return sendMessage(parentConnection, pageIdentifier->toUInt64());
}

template<typename U> static bool sendNotificationMessage(U&& message, WebPage* page)
{
    return sendMessage(page, [&] (auto& connection, auto destinationIdentifier) {
        return connection.send(std::forward<U>(message), destinationIdentifier) == IPC::Error::NoError;
    });
}

template<typename U, typename C>
static bool sendNotificationMessageWithAsyncReply(U&& message, WebPage* page, C&& callback)
{
    return sendMessage(page, [&] (auto& connection, auto destinationIdentifier) {
        return !!connection.sendWithAsyncReply(std::forward<U>(message), WTFMove(callback), destinationIdentifier);
    });
}
#endif // ENABLE(NOTIFICATIONS)

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebNotificationManager);

ASCIILiteral WebNotificationManager::supplementName()
{
    return "WebNotificationManager"_s;
}

WebNotificationManager::WebNotificationManager(WebProcess& process)
    : m_process(process)
{
#if ENABLE(NOTIFICATIONS)
    process.addMessageReceiver(Messages::WebNotificationManager::messageReceiverName(), *this);
#endif
}

WebNotificationManager::~WebNotificationManager() = default;

void WebNotificationManager::ref() const
{
    m_process->ref();
}

void WebNotificationManager::deref() const
{
    m_process->deref();
}

void WebNotificationManager::initialize(const WebProcessCreationParameters& parameters)
{
#if ENABLE(NOTIFICATIONS)
    m_permissionsMap = parameters.notificationPermissions;
#else
    UNUSED_PARAM(parameters);
#endif
}

void WebNotificationManager::didUpdateNotificationDecision(const String& originString, bool allowed)
{
#if ENABLE(NOTIFICATIONS)
    if (!originString.isEmpty())
        m_permissionsMap.set(originString, allowed);
#else
    UNUSED_PARAM(originString);
    UNUSED_PARAM(allowed);
#endif
}

void WebNotificationManager::didRemoveNotificationDecisions(const Vector<String>& originStrings)
{
#if ENABLE(NOTIFICATIONS)
    for (auto& originString : originStrings) {
        if (!originString.isEmpty())
            m_permissionsMap.remove(originString);
    }
#else
    UNUSED_PARAM(originStrings);
#endif
}

NotificationClient::Permission WebNotificationManager::policyForOrigin(const String& originString, WebPage* page) const
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS) && ENABLE(NOTIFICATIONS)
    if (DeprecatedGlobalSettings::builtInNotificationsEnabled()) {
        Ref connection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
        auto origin = SecurityOriginData::fromURL(URL { originString });
        auto result = connection->sendSync(Messages::NotificationManagerMessageHandler::GetPermissionStateSync(WTFMove(origin)), WebProcess::singleton().sessionID().toUInt64());
        if (!result.succeeded())
            RELEASE_LOG_ERROR(Notifications, "Could not look up notification permission for origin %" SENSITIVE_LOG_STRING": %u", originString.utf8().data(), static_cast<unsigned>(result.error()));

        auto [pushPermission] = result.takeReplyOr(PushPermissionState::Denied);
        switch (pushPermission) {
        case PushPermissionState::Denied:
            return NotificationPermission::Denied;
        case PushPermissionState::Granted:
            return NotificationPermission::Granted;
        case PushPermissionState::Prompt:
            return NotificationPermission::Default;
        default:
            RELEASE_ASSERT_NOT_REACHED();
        }
    }
#endif // ENABLE(WEB_PUSH_NOTIFICATIONS) && ENABLE(NOTIFICATIONS)

#if ENABLE(NOTIFICATIONS)
    if (originString.isEmpty())
        return NotificationClient::Permission::Default;

    auto it = m_permissionsMap.find(originString);
    if (it != m_permissionsMap.end()) {
        if (it->value && page)
            sendNotificationMessage(Messages::NotificationManagerMessageHandler::PageWasNotifiedOfNotificationPermission(), page);
        return it->value ? NotificationClient::Permission::Granted : NotificationClient::Permission::Denied;
    }
#else
    UNUSED_PARAM(originString);
#endif
    
    return NotificationClient::Permission::Default;
}

void WebNotificationManager::removeAllPermissionsForTesting()
{
#if ENABLE(NOTIFICATIONS)
    m_permissionsMap.clear();
#endif
}

bool WebNotificationManager::show(NotificationData&& notification, RefPtr<NotificationResources>&& resources, WebPage* page, CompletionHandler<void()>&& callback)
{
#if ENABLE(NOTIFICATIONS)
    auto notificationID = notification.notificationID;
    LOG(Notifications, "WebProcess %i going to show notification %s", getpid(), notificationID.toString().utf8().data());

    ASSERT(isMainRunLoop());
    if (page && !page->corePage()->settings().notificationsEnabled()) {
        callback();
        return false;
    }

    if (!sendNotificationMessageWithAsyncReply(Messages::NotificationManagerMessageHandler::ShowNotification(notification, resources), page, WTFMove(callback)))
        return false;

    if (!notification.isPersistent()) {
        ASSERT(!m_nonPersistentNotificationsContexts.contains(notificationID));
        RELEASE_ASSERT(notification.contextIdentifier);
        m_nonPersistentNotificationsContexts.add(notificationID, *notification.contextIdentifier);
    }
    return true;
#else
    UNUSED_PARAM(notification);
    UNUSED_PARAM(resources);
    UNUSED_PARAM(page);
    return false;
#endif
}

void WebNotificationManager::cancel(NotificationData&& notification, WebPage* page)
{
    ASSERT(isMainRunLoop());

#if ENABLE(NOTIFICATIONS)
    auto identifier = notification.notificationID;
    ASSERT(notification.isPersistent() || m_nonPersistentNotificationsContexts.contains(identifier));

    auto origin = WebCore::SecurityOriginData::fromURL(URL { notification.originString });
    if (!sendNotificationMessage(Messages::NotificationManagerMessageHandler::CancelNotification(WTFMove(origin), identifier), page))
        return;
#else
    UNUSED_PARAM(notification);
    UNUSED_PARAM(page);
#endif
}

void WebNotificationManager::requestPermission(WebCore::SecurityOriginData&& origin, RefPtr<WebPage> page, CompletionHandler<void(bool)>&& callback)
{
    ASSERT(isMainRunLoop());

#if ENABLE(NOTIFICATIONS)
    sendNotificationMessageWithAsyncReply(Messages::NotificationManagerMessageHandler::RequestPermission(WTFMove(origin)), page.get(), WTFMove(callback));
#else
    UNUSED_PARAM(origin);
    callback(false);
#endif
}

void WebNotificationManager::didDestroyNotification(NotificationData&& notification, WebPage* page)
{
    ASSERT(isMainRunLoop());

#if ENABLE(NOTIFICATIONS)
    auto identifier = notification.notificationID;
    if (!notification.isPersistent())
        m_nonPersistentNotificationsContexts.remove(identifier);

    sendNotificationMessage(Messages::NotificationManagerMessageHandler::DidDestroyNotification(identifier), page);
#else
    UNUSED_PARAM(notification);
    UNUSED_PARAM(page);
#endif
}

void WebNotificationManager::didShowNotification(const WTF::UUID& notificationID)
{
    ASSERT(isMainRunLoop());

    LOG(Notifications, "WebProcess %i DID SHOW notification %s", getpid(), notificationID.toString().utf8().data());

#if ENABLE(NOTIFICATIONS)
    auto contextIdentifier = m_nonPersistentNotificationsContexts.get(notificationID);
    if (!contextIdentifier)
        return;

    Notification::ensureOnNotificationThread(contextIdentifier, notificationID, [](auto* notification) {
        if (notification)
            notification->dispatchShowEvent();
    });
#else
    UNUSED_PARAM(notificationID);
#endif
}

void WebNotificationManager::didClickNotification(const WTF::UUID& notificationID)
{
    ASSERT(isMainRunLoop());

    LOG(Notifications, "WebProcess %i DID CLICK notification %s", getpid(), notificationID.toString().utf8().data());

#if ENABLE(NOTIFICATIONS)
    auto contextIdentifier = m_nonPersistentNotificationsContexts.get(notificationID);
    if (!contextIdentifier)
        return;

    LOG(Notifications, "WebProcess %i handling click event for notification %s", getpid(), notificationID.toString().utf8().data());

    Notification::ensureOnNotificationThread(contextIdentifier, notificationID, [](auto* notification) {
        if (!notification)
            return;

        // Indicate that this event is being dispatched in reaction to a user's interaction with a platform notification.
        if (isMainRunLoop()) {
            UserGestureIndicator indicator(IsProcessingUserGesture::Yes);
            notification->dispatchClickEvent();
        } else
            notification->dispatchClickEvent();
    });
#else
    UNUSED_PARAM(notificationID);
#endif
}

void WebNotificationManager::didCloseNotifications(const Vector<WTF::UUID>& notificationIDs)
{
    ASSERT(isMainRunLoop());

#if ENABLE(NOTIFICATIONS)
    for (auto& notificationID : notificationIDs) {
        auto contextIdentifier = m_nonPersistentNotificationsContexts.get(notificationID);
        if (!contextIdentifier)
            continue;

        Notification::ensureOnNotificationThread(contextIdentifier, notificationID, [](auto* notification) {
            if (notification)
                notification->dispatchCloseEvent();
        });
    }
#else
    UNUSED_PARAM(notificationIDs);
#endif
}

} // namespace WebKit
