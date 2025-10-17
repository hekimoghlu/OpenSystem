/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include "SharedWorkerContextManager.h"

#include "Logging.h"
#include "SharedWorkerGlobalScope.h"
#include "SharedWorkerThread.h"
#include "SharedWorkerThreadProxy.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SharedWorkerContextManager);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SharedWorkerContextManager::Connection);

SharedWorkerContextManager& SharedWorkerContextManager::singleton()
{
    static NeverDestroyed<SharedWorkerContextManager> sharedManager;
    return sharedManager;
}

SharedWorkerThreadProxy* SharedWorkerContextManager::sharedWorker(SharedWorkerIdentifier sharedWorkerIdentifier) const
{
    return m_workerMap.get(sharedWorkerIdentifier);
}

void SharedWorkerContextManager::stopSharedWorker(SharedWorkerIdentifier sharedWorkerIdentifier)
{
    auto worker = m_workerMap.take(sharedWorkerIdentifier);
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::stopSharedWorker: sharedWorkerIdentifier=%" PRIu64 ", worker=%p", sharedWorkerIdentifier.toUInt64(), worker.get());
    if (!worker)
        return;

    worker->setAsTerminatingOrTerminated();

    // FIXME: We should be able to deal with the thread being unresponsive here.

    auto& thread = worker->thread();
    thread.stop([worker = WTFMove(worker)]() mutable {
        // Spin the runloop before releasing the shared worker thread proxy, as there would otherwise be
        // a race towards its destruction.
        callOnMainThread([worker = WTFMove(worker)] { });
    });

    if (auto* connection = SharedWorkerContextManager::singleton().connection())
        connection->sharedWorkerTerminated(sharedWorkerIdentifier);
}

void SharedWorkerContextManager::suspendSharedWorker(SharedWorkerIdentifier sharedWorkerIdentifier)
{
    auto* worker = m_workerMap.get(sharedWorkerIdentifier);
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::suspendSharedWorker: sharedWorkerIdentifier=%" PRIu64 ", worker=%p", sharedWorkerIdentifier.toUInt64(), worker);
    if (worker)
        worker->thread().suspend();
}

void SharedWorkerContextManager::resumeSharedWorker(SharedWorkerIdentifier sharedWorkerIdentifier)
{
    auto* worker = m_workerMap.get(sharedWorkerIdentifier);
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::resumeSharedWorker: sharedWorkerIdentifier=%" PRIu64 ", worker=%p", sharedWorkerIdentifier.toUInt64(), worker);
    if (worker)
        worker->thread().resume();
}

void SharedWorkerContextManager::stopAllSharedWorkers()
{
    while (!m_workerMap.isEmpty())
        stopSharedWorker(m_workerMap.begin()->key);
}

void SharedWorkerContextManager::setConnection(RefPtr<Connection>&& connection)
{
    ASSERT(!m_connection || m_connection->isClosed());
    m_connection = WTFMove(connection);
}

auto SharedWorkerContextManager::connection() const -> Connection*
{
    return m_connection.get();
}

void SharedWorkerContextManager::registerSharedWorkerThread(Ref<SharedWorkerThreadProxy>&& proxy)
{
    ASSERT(isMainThread());
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::registerSharedWorkerThread: sharedWorkerIdentifier=%" PRIu64, proxy->identifier().toUInt64());

    auto result = m_workerMap.add(proxy->identifier(), proxy.copyRef());
    ASSERT_UNUSED(result, result.isNewEntry);

    proxy->thread().start([](const String& /*exceptionMessage*/) { });
}

void SharedWorkerContextManager::Connection::postConnectEvent(SharedWorkerIdentifier sharedWorkerIdentifier, TransferredMessagePort&& transferredPort, String&& sourceOrigin, CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainThread());
    auto* proxy = SharedWorkerContextManager::singleton().sharedWorker(sharedWorkerIdentifier);
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::Connection::postConnectEvent: sharedWorkerIdentifier=%" PRIu64 ", proxy=%p", sharedWorkerIdentifier.toUInt64(), proxy);
    if (!proxy)
        return completionHandler(false);

    proxy->thread().runLoop().postTask([transferredPort = WTFMove(transferredPort), sourceOrigin = WTFMove(sourceOrigin).isolatedCopy()] (auto& scriptExecutionContext) mutable {
        ASSERT(!isMainThread());
        downcast<SharedWorkerGlobalScope>(scriptExecutionContext).postConnectEvent(WTFMove(transferredPort), WTFMove(sourceOrigin));
    });
    completionHandler(true);
}

void SharedWorkerContextManager::Connection::terminateSharedWorker(SharedWorkerIdentifier sharedWorkerIdentifier)
{
    ASSERT(isMainThread());
    RELEASE_LOG(SharedWorker, "SharedWorkerContextManager::Connection::terminateSharedWorker: sharedWorkerIdentifier=%" PRIu64, sharedWorkerIdentifier.toUInt64());
    SharedWorkerContextManager::singleton().stopSharedWorker(sharedWorkerIdentifier);
}

void SharedWorkerContextManager::Connection::suspendSharedWorker(SharedWorkerIdentifier identifier)
{
    SharedWorkerContextManager::singleton().suspendSharedWorker(identifier);
}

void SharedWorkerContextManager::Connection::resumeSharedWorker(SharedWorkerIdentifier identifier)
{
    SharedWorkerContextManager::singleton().resumeSharedWorker(identifier);
}

void SharedWorkerContextManager::forEachSharedWorker(const Function<Function<void(ScriptExecutionContext&)>()>& createTask)
{
    for (auto& worker : m_workerMap.values())
        worker->thread().runLoop().postTask(createTask());
}

} // namespace WebCore
