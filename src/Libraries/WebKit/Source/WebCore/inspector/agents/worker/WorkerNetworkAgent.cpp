/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#include "WorkerNetworkAgent.h"

#include "SharedBuffer.h"
#include "WorkerDebuggerProxy.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerThread.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerNetworkAgent);

WorkerNetworkAgent::WorkerNetworkAgent(WorkerAgentContext& context)
    : InspectorNetworkAgent(context)
    , m_globalScope(context.globalScope)
{
    ASSERT(context.globalScope->isContextThread());
}

WorkerNetworkAgent::~WorkerNetworkAgent() = default;

Inspector::Protocol::Network::LoaderId WorkerNetworkAgent::loaderIdentifier(DocumentLoader*)
{
    return { };
}

Inspector::Protocol::Network::FrameId WorkerNetworkAgent::frameIdentifier(DocumentLoader*)
{
    return { };
}

Vector<WebSocket*> WorkerNetworkAgent::activeWebSockets()
{
    // FIXME: <https://webkit.org/b/168475> Web Inspector: Correctly display worker's WebSockets
    return { };
}

void WorkerNetworkAgent::setResourceCachingDisabledInternal(bool disabled)
{
    if (auto* workerDebuggerProxy = m_globalScope.workerOrWorkletThread()->workerDebuggerProxy())
        workerDebuggerProxy->setResourceCachingDisabledByWebInspector(disabled);
}

#if ENABLE(INSPECTOR_NETWORK_THROTTLING)

bool WorkerNetworkAgent::setEmulatedConditionsInternal(std::optional<int>&& /* bytesPerSecondLimit */)
{
    return false;
}

#endif // ENABLE(INSPECTOR_NETWORK_THROTTLING)

ScriptExecutionContext* WorkerNetworkAgent::scriptExecutionContext(Inspector::Protocol::ErrorString&, const Inspector::Protocol::Network::FrameId&)
{
    return &m_globalScope;
}

void WorkerNetworkAgent::addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&& message)
{
    m_globalScope.addConsoleMessage(WTFMove(message));
}

} // namespace WebCore
