/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "WebNotificationClient.h"

#if ENABLE(NOTIFICATIONS)

#include "NotificationPermissionRequestManager.h"
#include "WebNotificationManager.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/NotificationData.h>
#include <WebCore/Page.h>
#include <WebCore/ScriptExecutionContext.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebNotificationClient);

WebNotificationClient::WebNotificationClient(WebPage* page)
    : m_page(page)
{
    ASSERT(isMainRunLoop());
}

WebNotificationClient::~WebNotificationClient()
{
    ASSERT(isMainRunLoop());
}

bool WebNotificationClient::show(ScriptExecutionContext& context, NotificationData&& notification, RefPtr<NotificationResources>&& resources, CompletionHandler<void()>&& callback)
{
    bool result;
    callOnMainRunLoopAndWait([&result, notification = WTFMove(notification).isolatedCopy(), resources = WTFMove(resources), page = m_page, contextIdentifier = context.identifier(), callbackIdentifier = context.addNotificationCallback(WTFMove(callback))]() mutable {
        result = WebProcess::singleton().supplement<WebNotificationManager>()->show(WTFMove(notification), WTFMove(resources), RefPtr { page.get() }.get(), [contextIdentifier, callbackIdentifier] {
            ScriptExecutionContext::ensureOnContextThread(contextIdentifier, [callbackIdentifier](auto& context) {
                if (auto callback = context.takeNotificationCallback(callbackIdentifier))
                    callback();
            });
        });
    });
    return result;
}

void WebNotificationClient::cancel(NotificationData&& notification)
{
    callOnMainRunLoopAndWait([notification = WTFMove(notification).isolatedCopy(), page = m_page]() mutable {
        WebProcess::singleton().supplement<WebNotificationManager>()->cancel(WTFMove(notification), RefPtr { page.get() }.get());
    });
}

void WebNotificationClient::notificationObjectDestroyed(NotificationData&& notification)
{
    callOnMainRunLoopAndWait([notification = WTFMove(notification).isolatedCopy(), page = m_page]() mutable {
        WebProcess::singleton().supplement<WebNotificationManager>()->didDestroyNotification(WTFMove(notification), RefPtr { page.get() }.get());
    });
}

void WebNotificationClient::notificationControllerDestroyed()
{
    callOnMainRunLoop([this] {
        delete this;
    });
}

void WebNotificationClient::requestPermission(ScriptExecutionContext& context, PermissionHandler&& permissionHandler)
{
    // Only Window clients can request permission
    ASSERT(isMainRunLoop());
    ASSERT(m_page);

    if (!isMainRunLoop() || !context.isDocument() || WebProcess::singleton().sessionID().isEphemeral())
        return permissionHandler(NotificationClient::Permission::Denied);

    RefPtr securityOrigin = context.securityOrigin();
    if (!securityOrigin)
        return permissionHandler(NotificationClient::Permission::Denied);

    // Add origin to list of origins that have requested permission to use the Notifications API.
    m_notificationPermissionRequesters.add(securityOrigin->data());

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    if (DeprecatedGlobalSettings::builtInNotificationsEnabled()) {
        auto handler = [permissionHandler = WTFMove(permissionHandler)](bool granted) mutable {
            permissionHandler(granted ? NotificationPermission::Granted : NotificationPermission::Denied);
        };
        WebProcess::singleton().supplement<WebNotificationManager>()->requestPermission(WebCore::SecurityOriginData { securityOrigin->data() }, RefPtr { m_page.get() }, WTFMove(handler));
        return;
    }
#endif

    Ref { *m_page }->notificationPermissionRequestManager()->startRequest(securityOrigin->data(), WTFMove(permissionHandler));
}

NotificationClient::Permission WebNotificationClient::checkPermission(ScriptExecutionContext* context)
{
    if (!context || (!context->isDocument() && !context->isServiceWorkerGlobalScope()))
        return NotificationClient::Permission::Denied;

    RefPtr origin = context->securityOrigin();
    if (!origin)
        return NotificationClient::Permission::Denied;

    bool hasRequestedPermission = m_notificationPermissionRequesters.contains(origin->data());
    if (WebProcess::singleton().sessionID().isEphemeral())
        return hasRequestedPermission ? NotificationClient::Permission::Denied : NotificationClient::Permission::Default;

    NotificationClient::Permission resultPermission;
    if (RefPtr document = dynamicDowncast<Document>(*context)) {
        ASSERT(isMainRunLoop());
        RefPtr page = document->page() ? WebPage::fromCorePage(*document->page()) : nullptr;
        resultPermission = WebProcess::singleton().supplement<WebNotificationManager>()->policyForOrigin(origin->data().toString(), page.get());
    } else {
        callOnMainRunLoopAndWait([&resultPermission, origin = origin->data().toString().isolatedCopy()] {
            resultPermission = WebProcess::singleton().supplement<WebNotificationManager>()->policyForOrigin(origin);
        });
    }

    // To reduce fingerprinting, if the origin has not requested permission to use the
    // Notifications API, and the permission state is "denied", return "default" instead.
    if (resultPermission == NotificationClient::Permission::Denied && !hasRequestedPermission)
        return NotificationClient::Permission::Default;

    return resultPermission;
}

void WebNotificationClient::clearNotificationPermissionState()
{
    m_notificationPermissionRequesters.clear();
}

} // namespace WebKit

#endif // ENABLE(NOTIFICATIONS)
