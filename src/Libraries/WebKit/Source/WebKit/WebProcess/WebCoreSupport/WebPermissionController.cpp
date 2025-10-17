/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#include "WebPermissionController.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebPermissionControllerMessages.h"
#include "WebPermissionControllerProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/Document.h>
#include <WebCore/Page.h>
#include <WebCore/PermissionObserver.h>
#include <WebCore/PermissionQuerySource.h>
#include <WebCore/PermissionState.h>
#include <WebCore/Permissions.h>
#include <WebCore/SecurityOriginData.h>
#include <optional>

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
#include "NetworkProcessConnection.h"
#include "NotificationManagerMessageHandlerMessages.h"
#include <WebCore/PushPermissionState.h>
#endif

namespace WebKit {

Ref<WebPermissionController> WebPermissionController::create(WebProcess& process)
{
    return adoptRef(*new WebPermissionController(process));
}

WebPermissionController::WebPermissionController(WebProcess& process)
{
    process.addMessageReceiver(Messages::WebPermissionController::messageReceiverName(), *this);
}

WebPermissionController::~WebPermissionController()
{
    WebProcess::singleton().removeMessageReceiver(Messages::WebPermissionController::messageReceiverName());
}

void WebPermissionController::query(WebCore::ClientOrigin&& origin, WebCore::PermissionDescriptor descriptor, const WeakPtr<WebCore::Page>& page, WebCore::PermissionQuerySource source, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&& completionHandler)
{
#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    if (DeprecatedGlobalSettings::builtInNotificationsEnabled() && (descriptor.name == PermissionName::Notifications || descriptor.name == PermissionName::Push)) {
        Ref connection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
        auto notificationPermissionHandler = [completionHandler = WTFMove(completionHandler)](WebCore::PushPermissionState pushPermissionState) mutable {
            auto state = [pushPermissionState]() -> WebCore::PermissionState {
                switch (pushPermissionState) {
                case WebCore::PushPermissionState::Granted: return WebCore::PermissionState::Granted;
                case WebCore::PushPermissionState::Denied: return WebCore::PermissionState::Denied;
                case WebCore::PushPermissionState::Prompt: return WebCore::PermissionState::Prompt;
                default: RELEASE_ASSERT_NOT_REACHED();
                }
            }();
            completionHandler(state);
        };
        connection->sendWithAsyncReply(Messages::NotificationManagerMessageHandler::GetPermissionState(origin.clientOrigin), WTFMove(notificationPermissionHandler), WebProcess::singleton().sessionID().toUInt64());
        return;
    }
#endif

    std::optional<WebPageProxyIdentifier> proxyIdentifier;
    if (source == WebCore::PermissionQuerySource::Window || source == WebCore::PermissionQuerySource::DedicatedWorker) {
        ASSERT(page);
        proxyIdentifier = WebPage::fromCorePage(*page)->webPageProxyIdentifier();
    }

    WebProcess::singleton().sendWithAsyncReply(Messages::WebPermissionControllerProxy::Query(origin, descriptor, proxyIdentifier, source), WTFMove(completionHandler));
}

void WebPermissionController::addObserver(WebCore::PermissionObserver& observer)
{
    m_observers.add(observer);
}

void WebPermissionController::removeObserver(WebCore::PermissionObserver& observer)
{
    m_observers.remove(observer);
}

void WebPermissionController::permissionChanged(WebCore::PermissionName permissionName, const WebCore::SecurityOriginData& topOrigin)
{
    ASSERT(isMainRunLoop());

    for (auto& observer : m_observers) {
        if (observer.descriptor().name != permissionName || observer.origin().topOrigin != topOrigin)
            return;

        auto source = observer.source();
        if (!observer.page() && (source == WebCore::PermissionQuerySource::Window || source == WebCore::PermissionQuerySource::DedicatedWorker))
            return;

        query(WebCore::ClientOrigin { observer.origin() }, WebCore::PermissionDescriptor { permissionName }, observer.page(), source, [observer = WeakPtr { observer }](auto newState) {
            if (observer && newState != observer->currentState())
                observer->stateChanged(*newState);
        });
    }
}

} // namespace WebKit
