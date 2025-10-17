/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "ServiceWorkerInspectorProxy.h"

#include "SWContextManager.h"
#include "ScriptExecutionContext.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerThreadProxy.h"
#include "WorkerInspectorController.h"
#include "WorkerRunLoop.h"
#include <JavaScriptCore/InspectorAgentBase.h>
#include <JavaScriptCore/InspectorFrontendChannel.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerInspectorProxy);

ServiceWorkerInspectorProxy::ServiceWorkerInspectorProxy(ServiceWorkerThreadProxy& serviceWorkerThreadProxy)
    : m_serviceWorkerThreadProxy(serviceWorkerThreadProxy)
{
    ASSERT(isMainThread());
}

ServiceWorkerInspectorProxy::~ServiceWorkerInspectorProxy()
{
    ASSERT(isMainThread());
    ASSERT(!m_channel);
}

void ServiceWorkerInspectorProxy::serviceWorkerTerminated()
{
    m_channel = nullptr;
}

void ServiceWorkerInspectorProxy::connectToWorker(FrontendChannel& channel)
{
    m_channel = &channel;

    RefPtr serviceWorkerThreadProxy = m_serviceWorkerThreadProxy.get();
    SWContextManager::singleton().setAsInspected(serviceWorkerThreadProxy->identifier(), true);
    serviceWorkerThreadProxy->thread().runLoop().postDebuggerTask([] (ScriptExecutionContext& context) {
        downcast<WorkerGlobalScope>(context).inspectorController().connectFrontend();
    });
}

void ServiceWorkerInspectorProxy::disconnectFromWorker(FrontendChannel& channel)
{
    ASSERT_UNUSED(channel, &channel == m_channel);
    m_channel = nullptr;

    RefPtr serviceWorkerThreadProxy = m_serviceWorkerThreadProxy.get();
    SWContextManager::singleton().setAsInspected(serviceWorkerThreadProxy->identifier(), false);
    serviceWorkerThreadProxy->thread().runLoop().postDebuggerTask([] (ScriptExecutionContext& context) {
        downcast<WorkerGlobalScope>(context).inspectorController().disconnectFrontend(DisconnectReason::InspectorDestroyed);

        // In case the worker is paused running debugger tasks, ensure we break out of
        // the pause since this will be the last debugger task we send to the worker.
        downcast<WorkerGlobalScope>(context).thread().stopRunningDebuggerTasks();
    });
}

void ServiceWorkerInspectorProxy::sendMessageToWorker(String&& message)
{
    m_serviceWorkerThreadProxy.get()->thread().runLoop().postDebuggerTask([message = WTFMove(message).isolatedCopy()] (ScriptExecutionContext& context) {
        downcast<WorkerGlobalScope>(context).inspectorController().dispatchMessageFromFrontend(message);
    });
}

void ServiceWorkerInspectorProxy::sendMessageFromWorkerToFrontend(String&& message)
{
    if (!m_channel)
        return;

    m_channel->sendMessageToFrontend(WTFMove(message));
}

} // namespace WebCore
