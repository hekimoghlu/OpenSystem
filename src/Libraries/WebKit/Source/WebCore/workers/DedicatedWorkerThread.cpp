/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "DedicatedWorkerThread.h"

#include "DedicatedWorkerGlobalScope.h"
#include "SecurityOrigin.h"
#include "ServiceWorker.h"
#include "WorkerObjectProxy.h"

namespace WebCore {

DedicatedWorkerThread::DedicatedWorkerThread(const WorkerParameters& params, const ScriptBuffer& sourceCode, WorkerLoaderProxy& workerLoaderProxy, WorkerDebuggerProxy& workerDebuggerProxy, WorkerObjectProxy& workerObjectProxy, WorkerBadgeProxy& workerBadgeProxy, WorkerThreadStartMode startMode, const SecurityOrigin& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, JSC::RuntimeFlags runtimeFlags)
    : WorkerThread(params, sourceCode, workerLoaderProxy, workerDebuggerProxy, workerObjectProxy, workerBadgeProxy, startMode, topOrigin, connectionProxy, socketProvider, runtimeFlags)
    , m_workerObjectProxy(workerObjectProxy)
{
}

DedicatedWorkerThread::~DedicatedWorkerThread() = default;

Ref<WorkerGlobalScope> DedicatedWorkerThread::createWorkerGlobalScope(const WorkerParameters& params, Ref<SecurityOrigin>&& origin, Ref<SecurityOrigin>&& topOrigin)
{
    auto scope = DedicatedWorkerGlobalScope::create(params, WTFMove(origin), *this, WTFMove(topOrigin), idbConnectionProxy(), socketProvider(), WTFMove(m_workerClient));
    if (params.serviceWorkerData)
        scope->setActiveServiceWorker(ServiceWorker::getOrCreate(scope.get(), ServiceWorkerData { *params.serviceWorkerData }));
    scope->updateServiceWorkerClientData();
    return scope;
}

} // namespace WebCore
