/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include "NotificationPermissionRequestManager.h"

#include "MessageSenderInlines.h"
#include "NotificationManagerMessageHandlerMessages.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/Notification.h>
#include <WebCore/Page.h>
#include <WebCore/ScriptExecutionContext.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/Settings.h>

#if ENABLE(NOTIFICATIONS)
#include "WebNotificationManager.h"
#endif

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
#include "NetworkProcessConnection.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#endif

namespace WebKit {
using namespace WebCore;

Ref<NotificationPermissionRequestManager> NotificationPermissionRequestManager::create(WebPage* page)
{
    return adoptRef(*new NotificationPermissionRequestManager(page));
}

#if ENABLE(NOTIFICATIONS)
NotificationPermissionRequestManager::NotificationPermissionRequestManager(WebPage* page)
    : m_page(page)
{
}
#else
NotificationPermissionRequestManager::NotificationPermissionRequestManager(WebPage*)
{
}
#endif

NotificationPermissionRequestManager::~NotificationPermissionRequestManager()
{
#if ENABLE(NOTIFICATIONS)
    auto requestsPerOrigin = std::exchange(m_requestsPerOrigin, { });
    for (auto& permissionHandlers : requestsPerOrigin.values())
        callPermissionHandlersWith(permissionHandlers, Permission::Denied);
#endif
}

#if ENABLE(NOTIFICATIONS)
void NotificationPermissionRequestManager::startRequest(const SecurityOriginData& securityOrigin, PermissionHandler&& permissionHandler)
{
    auto addResult = m_requestsPerOrigin.add(securityOrigin, PermissionHandlers { });
    addResult.iterator->value.append(WTFMove(permissionHandler));
    if (!addResult.isNewEntry)
        return;

    m_page->sendWithAsyncReply(Messages::WebPageProxy::RequestNotificationPermission(securityOrigin.toString()), [this, protectedThis = Ref { *this }, securityOrigin, permissionHandler = WTFMove(permissionHandler)](bool allowed) mutable {

        auto innerPermissionHandler = [this, protectedThis = Ref { *this }, securityOrigin, permissionHandler = WTFMove(permissionHandler)] (bool allowed) mutable {
            WebProcess::singleton().protectedNotificationManager()->didUpdateNotificationDecision(securityOrigin.toString(), allowed);

            auto permissionHandlers = m_requestsPerOrigin.take(securityOrigin);
            callPermissionHandlersWith(permissionHandlers, allowed ? Permission::Granted : Permission::Denied);
        };

        innerPermissionHandler(allowed);
    });
}

void NotificationPermissionRequestManager::callPermissionHandlersWith(PermissionHandlers& permissionHandlers, Permission permission)
{
    for (auto& permissionHandler : permissionHandlers)
        permissionHandler(permission);
}
#endif

auto NotificationPermissionRequestManager::permissionLevel(const SecurityOriginData& securityOrigin) -> Permission
{
#if ENABLE(NOTIFICATIONS)
    if (!m_page->corePage()->settings().notificationsEnabled())
        return Permission::Denied;
    
    return WebProcess::singleton().protectedNotificationManager()->policyForOrigin(securityOrigin.toString());
#else
    UNUSED_PARAM(securityOrigin);
    return Permission::Denied;
#endif
}

void NotificationPermissionRequestManager::setPermissionLevelForTesting(const String& originString, bool allowed)
{
#if ENABLE(NOTIFICATIONS)
    WebProcess::singleton().protectedNotificationManager()->didUpdateNotificationDecision(originString, allowed);
#else
    UNUSED_PARAM(originString);
    UNUSED_PARAM(allowed);
#endif
}

void NotificationPermissionRequestManager::removeAllPermissionsForTesting()
{
#if ENABLE(NOTIFICATIONS)
    WebProcess::singleton().protectedNotificationManager()->removeAllPermissionsForTesting();
#endif
}

} // namespace WebKit
