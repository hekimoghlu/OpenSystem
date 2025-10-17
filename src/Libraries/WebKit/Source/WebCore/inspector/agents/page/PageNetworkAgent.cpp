/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#include "PageNetworkAgent.h"

#include "Document.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "InspectorClient.h"
#include "InstrumentingAgents.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include "ThreadableWebSocketChannel.h"
#include "WebSocket.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageNetworkAgent);

PageNetworkAgent::PageNetworkAgent(PageAgentContext& context, InspectorClient* client)
    : InspectorNetworkAgent(context)
    , m_inspectedPage(context.inspectedPage)
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    , m_client(client)
#endif
{
#if !ENABLE(INSPECTOR_NETWORK_THROTTLING)
    UNUSED_PARAM(client);
#endif
}

PageNetworkAgent::~PageNetworkAgent() = default;

Inspector::Protocol::Network::LoaderId PageNetworkAgent::loaderIdentifier(DocumentLoader* loader)
{
    if (loader) {
        if (auto* pageAgent = m_instrumentingAgents.enabledPageAgent())
            return pageAgent->loaderId(loader);
    }
    return { };
}

Inspector::Protocol::Network::FrameId PageNetworkAgent::frameIdentifier(DocumentLoader* loader)
{
    if (loader) {
        if (auto* pageAgent = m_instrumentingAgents.enabledPageAgent())
            return pageAgent->frameId(loader->frame());
    }
    return { };
}

Vector<WebSocket*> PageNetworkAgent::activeWebSockets()
{
    Vector<WebSocket*> webSockets;

    for (auto* webSocket : WebSocket::allActiveWebSockets()) {
        auto channel = webSocket->channel();
        if (!channel)
            continue;

        if (!channel->hasCreatedHandshake())
            continue;

        RefPtr document = dynamicDowncast<Document>(webSocket->scriptExecutionContext());
        if (!document)
            continue;

        // FIXME: <https://webkit.org/b/168475> Web Inspector: Correctly display iframe's WebSockets
        if (document->page() != m_inspectedPage.ptr())
            continue;

        webSockets.append(webSocket);
    }

    return webSockets;
}

void PageNetworkAgent::setResourceCachingDisabledInternal(bool disabled)
{
    m_inspectedPage->setResourceCachingDisabledByWebInspector(disabled);
}

#if ENABLE(INSPECTOR_NETWORK_THROTTLING)

bool PageNetworkAgent::setEmulatedConditionsInternal(std::optional<int>&& bytesPerSecondLimit)
{
    return m_client && m_client->setEmulatedConditions(WTFMove(bytesPerSecondLimit));
}

#endif // ENABLE(INSPECTOR_NETWORK_THROTTLING)

ScriptExecutionContext* PageNetworkAgent::scriptExecutionContext(Inspector::Protocol::ErrorString& errorString, const Inspector::Protocol::Network::FrameId& frameId)
{
    auto* pageAgent = m_instrumentingAgents.enabledPageAgent();
    if (!pageAgent) {
        errorString = "Page domain must be enabled"_s;
        return nullptr;
    }

    auto* frame = pageAgent->assertFrame(errorString, frameId);
    if (!frame)
        return nullptr;

    auto* document = frame->document();
    if (!document) {
        errorString = "Missing frame of docuemnt for given frameId"_s;
        return nullptr;
    }

    return document;
}

void PageNetworkAgent::addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&& message)
{
    m_inspectedPage->console().addMessage(WTFMove(message));
}

} // namespace WebCore
