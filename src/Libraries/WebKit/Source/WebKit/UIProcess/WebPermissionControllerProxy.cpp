/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "WebPermissionControllerProxy.h"

#include "WebPageProxy.h"
#include "WebPermissionControllerProxyMessages.h"
#include "WebProcessProxy.h"
#include <WebCore/ClientOrigin.h>
#include <WebCore/PermissionDescriptor.h>
#include <WebCore/PermissionQuerySource.h>
#include <WebCore/PermissionState.h>
#include <WebCore/SecurityOriginData.h>
#include <optional>
#include <wtf/GetPtr.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPermissionControllerProxy);

WebPermissionControllerProxy::WebPermissionControllerProxy(WebProcessProxy& process)
    : m_process(process)
{
    protectedProcess()->addMessageReceiver(Messages::WebPermissionControllerProxy::messageReceiverName(), *this);
}

WebPermissionControllerProxy::~WebPermissionControllerProxy()
{
    protectedProcess()->removeMessageReceiver(Messages::WebPermissionControllerProxy::messageReceiverName());
}

void WebPermissionControllerProxy::ref() const
{
    m_process->ref();
}

void WebPermissionControllerProxy::deref() const
{
    m_process->deref();
}

Ref<WebProcessProxy> WebPermissionControllerProxy::protectedProcess() const
{
    return m_process.get();
}

void WebPermissionControllerProxy::query(const WebCore::ClientOrigin& clientOrigin, const WebCore::PermissionDescriptor& descriptor, std::optional<WebPageProxyIdentifier> identifier, WebCore::PermissionQuerySource source, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&& completionHandler)
{
    auto webPageProxy = identifier ? protectedProcess()->webPage(identifier.value()) : mostReasonableWebPageProxy(clientOrigin.topOrigin, source);

    if (!webPageProxy) {
        completionHandler(WebCore::PermissionState::Prompt);
        return;
    }

    webPageProxy->queryPermission(clientOrigin, descriptor, WTFMove(completionHandler));
}

RefPtr<WebPageProxy> WebPermissionControllerProxy::mostReasonableWebPageProxy(const WebCore::SecurityOriginData& topOrigin, WebCore::PermissionQuerySource source) const
{
    ASSERT(source == WebCore::PermissionQuerySource::SharedWorker || source == WebCore::PermissionQuerySource::ServiceWorker);
    
    RefPtr<WebPageProxy> webPageProxy;
    auto findWebPageProxy = [&topOrigin, &webPageProxy] (auto* processes) {
        if (!processes)
            return; 

        for (auto& process : *processes) {
            for (Ref potentialWebPageProxy : getPtr(process)->pages()) {
                if (WebCore::SecurityOriginData::fromURLWithoutStrictOpaqueness(URL { potentialWebPageProxy->currentURL() }) != topOrigin)
                    continue;
                // The most reasonable webPageProxy is the newest one (the one with the greatest identifier).
                if (webPageProxy && webPageProxy->identifier() > potentialWebPageProxy->identifier())
                    continue;
                webPageProxy = WTFMove(potentialWebPageProxy);
            }
        }
    };

    Vector<Ref<WebProcessProxy>> currentProcess { protectedProcess() };
    findWebPageProxy(&currentProcess);

    switch (source) {
    case WebCore::PermissionQuerySource::ServiceWorker:
        findWebPageProxy(protectedProcess()->serviceWorkerClientProcesses());
        break;
    case WebCore::PermissionQuerySource::SharedWorker:
        findWebPageProxy(protectedProcess()->sharedWorkerClientProcesses());
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }

    return webPageProxy;
}

} // namespace WebKit
