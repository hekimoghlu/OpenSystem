/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#include "ServiceWorkerAgent.h"

#include "SecurityOrigin.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerThread.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerAgent);

ServiceWorkerAgent::ServiceWorkerAgent(WorkerAgentContext& context)
    : InspectorAgentBase("ServiceWorker"_s, context)
    , m_serviceWorkerGlobalScope(downcast<ServiceWorkerGlobalScope>(context.globalScope.get()))
    , m_backendDispatcher(Inspector::ServiceWorkerBackendDispatcher::create(context.backendDispatcher, this))
{
    ASSERT(context.globalScope->isContextThread());
}

ServiceWorkerAgent::~ServiceWorkerAgent() = default;

void ServiceWorkerAgent::didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*)
{
}

void ServiceWorkerAgent::willDestroyFrontendAndBackend(Inspector::DisconnectReason)
{
}

Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::ServiceWorker::Configuration>> ServiceWorkerAgent::getInitializationInfo()
{
    return Inspector::Protocol::ServiceWorker::Configuration::create()
        .setTargetId(m_serviceWorkerGlobalScope->inspectorIdentifier())
        .setSecurityOrigin(m_serviceWorkerGlobalScope->securityOrigin()->toRawString())
        .setUrl(m_serviceWorkerGlobalScope->contextData().scriptURL.string())
        .setContent(m_serviceWorkerGlobalScope->contextData().script.toString())
        .release();
}

} // namespace WebCore
