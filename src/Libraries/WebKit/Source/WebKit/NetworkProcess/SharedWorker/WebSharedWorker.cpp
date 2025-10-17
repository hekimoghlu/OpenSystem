/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#include "WebSharedWorker.h"

#include "WebSharedWorkerServer.h"
#include "WebSharedWorkerServerToContextConnection.h"
#include <WebCore/Site.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakRef.h>

namespace WebKit {

static HashMap<WebCore::SharedWorkerIdentifier, WeakRef<WebSharedWorker>>& allWorkers()
{
    ASSERT(RunLoop::isMain());
    static NeverDestroyed<HashMap<WebCore::SharedWorkerIdentifier, WeakRef<WebSharedWorker>>> allWorkers;
    return allWorkers;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSharedWorker);


Ref<WebSharedWorker> WebSharedWorker::create(WebSharedWorkerServer& server, const WebCore::SharedWorkerKey& key, const WebCore::WorkerOptions& options)
{
    return adoptRef(*new WebSharedWorker(server, key, options));
}

WebSharedWorker::WebSharedWorker(WebSharedWorkerServer& server, const WebCore::SharedWorkerKey& key, const WebCore::WorkerOptions& workerOptions)
    : m_server(server)
    , m_key(key)
    , m_workerOptions(workerOptions)
{
    ASSERT(!allWorkers().contains(identifier()));
    allWorkers().add(identifier(), *this);
}

WebSharedWorker::~WebSharedWorker()
{
    if (RefPtr connection = contextConnection()) {
        for (auto& sharedWorkerObject : m_sharedWorkerObjects)
            connection->removeSharedWorkerObject(sharedWorkerObject.identifier);
    }

    ASSERT(allWorkers().get(identifier()) == this);
    allWorkers().remove(identifier());
}

WebSharedWorker* WebSharedWorker::fromIdentifier(WebCore::SharedWorkerIdentifier identifier)
{
    return allWorkers().get(identifier);
}

WebCore::RegistrableDomain WebSharedWorker::topRegistrableDomain() const
{
    return WebCore::RegistrableDomain { m_key.origin.topOrigin };
}

WebCore::Site WebSharedWorker::topSite() const
{
    return WebCore::Site { m_key.origin.topOrigin };
}

void WebSharedWorker::setFetchResult(WebCore::WorkerFetchResult&& fetchResult)
{
    m_fetchResult = WTFMove(fetchResult);
}

void WebSharedWorker::didCreateContextConnection(WebSharedWorkerServerToContextConnection& contextConnection)
{
    for (auto& sharedWorkerObject : m_sharedWorkerObjects)
        contextConnection.addSharedWorkerObject(sharedWorkerObject.identifier);
    if (didFinishFetching())
        launch(contextConnection);
}

void WebSharedWorker::launch(WebSharedWorkerServerToContextConnection& connection)
{
    connection.launchSharedWorker(*this);
    if (m_isSuspended)
        connection.suspendSharedWorker(identifier());
}

void WebSharedWorker::addSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, const WebCore::TransferredMessagePort& port)
{
    ASSERT(!m_sharedWorkerObjects.contains({ sharedWorkerObjectIdentifier, { false, port } }));
    m_sharedWorkerObjects.add({ sharedWorkerObjectIdentifier, { false, port } });
    if (RefPtr connection = contextConnection())
        connection->addSharedWorkerObject(sharedWorkerObjectIdentifier);

    resumeIfNeeded();
}

void WebSharedWorker::removeSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    m_sharedWorkerObjects.remove({ sharedWorkerObjectIdentifier, { } });
    if (RefPtr connection = contextConnection())
        connection->removeSharedWorkerObject(sharedWorkerObjectIdentifier);

    suspendIfNeeded();
}

void WebSharedWorker::suspend(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    auto iterator = m_sharedWorkerObjects.find({ sharedWorkerObjectIdentifier, { } });
    if (iterator == m_sharedWorkerObjects.end())
        return;

    iterator->state.isSuspended = true;
    ASSERT(!m_isSuspended);
    suspendIfNeeded();
}

void WebSharedWorker::suspendIfNeeded()
{
    if (m_isSuspended)
        return;

    for (auto& object : m_sharedWorkerObjects) {
        if (!object.state.isSuspended)
            return;
    }

    m_isSuspended = true;
    if (RefPtr connection = contextConnection())
        connection->suspendSharedWorker(identifier());
}

void WebSharedWorker::resume(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    auto iterator = m_sharedWorkerObjects.find({ sharedWorkerObjectIdentifier, { } });
    if (iterator == m_sharedWorkerObjects.end())
        return;

    iterator->state.isSuspended = false;
    resumeIfNeeded();
}

void WebSharedWorker::resumeIfNeeded()
{
    if (!m_isSuspended)
        return;

    m_isSuspended = false;
    if (RefPtr connection = contextConnection())
        connection->resumeSharedWorker(identifier());
}

void WebSharedWorker::forEachSharedWorkerObject(const Function<void(WebCore::SharedWorkerObjectIdentifier, const WebCore::TransferredMessagePort&)>& apply) const
{
    for (auto& object : m_sharedWorkerObjects)
        apply(object.identifier, *object.state.port);
}

std::optional<WebCore::ProcessIdentifier> WebSharedWorker::firstSharedWorkerObjectProcess() const
{
    if (m_sharedWorkerObjects.isEmpty())
        return std::nullopt;
    return m_sharedWorkerObjects.first().identifier.processIdentifier();
}

WebSharedWorkerServerToContextConnection* WebSharedWorker::contextConnection() const
{
    if (!m_server)
        return nullptr;
    return m_server->contextConnectionForRegistrableDomain(topRegistrableDomain());
}

} // namespace WebKit
