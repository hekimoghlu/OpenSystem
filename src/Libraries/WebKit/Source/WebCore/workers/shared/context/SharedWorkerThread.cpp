/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#include "SharedWorkerThread.h"

#include "Logging.h"
#include "ServiceWorker.h"
#include "SharedWorkerGlobalScope.h"
#include "WorkerObjectProxy.h"

namespace WebCore {

SharedWorkerThread::SharedWorkerThread(SharedWorkerIdentifier identifier, const WorkerParameters& parameters, const ScriptBuffer& sourceCode, WorkerLoaderProxy& loaderProxy, WorkerDebuggerProxy& debuggerProxy, WorkerObjectProxy& objectProxy, WorkerBadgeProxy& badgeProxy, WorkerThreadStartMode startMode, const SecurityOrigin& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, JSC::RuntimeFlags runtimeFlags)
    : WorkerThread(parameters, sourceCode, loaderProxy, debuggerProxy, objectProxy, badgeProxy, startMode, topOrigin, connectionProxy, socketProvider, runtimeFlags)
    , m_identifier(identifier)
    , m_name(parameters.name.isolatedCopy())
{
    RELEASE_LOG(SharedWorker, "%p - SharedWorkerThread::SharedWorkerThread: m_identifier=%" PRIu64, this, m_identifier.toUInt64());
}

Ref<WorkerGlobalScope> SharedWorkerThread::createWorkerGlobalScope(const WorkerParameters& parameters, Ref<SecurityOrigin>&& origin, Ref<SecurityOrigin>&& topOrigin)
{
    RELEASE_LOG(SharedWorker, "%p - SharedWorkerThread::createWorkerGlobalScope: m_identifier=%" PRIu64, this, m_identifier.toUInt64());
    auto scope = SharedWorkerGlobalScope::create(std::exchange(m_name, { }), parameters, WTFMove(origin), *this, WTFMove(topOrigin), idbConnectionProxy(), socketProvider(), WTFMove(m_workerClient));
    if (parameters.serviceWorkerData)
        scope->setActiveServiceWorker(ServiceWorker::getOrCreate(scope.get(), ServiceWorkerData { *parameters.serviceWorkerData }));
    scope->updateServiceWorkerClientData();
    return scope;
}

} // namespace WebCore
