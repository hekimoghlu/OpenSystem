/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#include "RemoteMediaResourceManager.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Connection.h"
#include "RemoteMediaResource.h"
#include "RemoteMediaResourceIdentifier.h"
#include "RemoteMediaResourceLoader.h"
#include "RemoteMediaResourceManagerMessages.h"
#include "SharedBufferReference.h"
#include <WebCore/PlatformMediaResourceLoader.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaResourceManager);

RemoteMediaResourceManager::RemoteMediaResourceManager()
{
}

RemoteMediaResourceManager::~RemoteMediaResourceManager()
{
    Locker locker { m_lock };
    // Shutdown any stale RemoteMediaResources. We must complete this step in a follow-up task to prevent re-entry in RemoteMediaResourceManager.
    callOnMainRunLoop([resources = WTFMove(m_remoteMediaResources)] {
        for (auto&& resource : resources) {
            if (RefPtr protectedResource = resource.value.get())
                protectedResource->shutdown();
        }
    });
}

void RemoteMediaResourceManager::stopListeningForIPC()
{
    assertIsMainThread();
    initializeConnection(nullptr);
}

void RemoteMediaResourceManager::initializeConnection(IPC::Connection* connection)
{
    assertIsMainThread();

    RefPtr protectedConnection = m_connection;
    if (protectedConnection == connection)
        return;

    if (protectedConnection)
        protectedConnection->removeWorkQueueMessageReceiver(Messages::RemoteMediaResourceManager::messageReceiverName());

    m_connection = connection;
    protectedConnection = m_connection;

    if (protectedConnection)
        protectedConnection->addWorkQueueMessageReceiver(Messages::RemoteMediaResourceManager::messageReceiverName(), RemoteMediaResourceLoader::defaultQueue(), *this);
}

void RemoteMediaResourceManager::addMediaResource(RemoteMediaResourceIdentifier remoteMediaResourceIdentifier, RemoteMediaResource& remoteMediaResource)
{
    Locker locker { m_lock };
    ASSERT(!m_remoteMediaResources.contains(remoteMediaResourceIdentifier));
    m_remoteMediaResources.add(remoteMediaResourceIdentifier, ThreadSafeWeakPtr { remoteMediaResource });
}

void RemoteMediaResourceManager::removeMediaResource(RemoteMediaResourceIdentifier remoteMediaResourceIdentifier)
{
    Locker locker { m_lock };
    ASSERT(m_remoteMediaResources.contains(remoteMediaResourceIdentifier));
    m_remoteMediaResources.remove(remoteMediaResourceIdentifier);
}

RefPtr<RemoteMediaResource> RemoteMediaResourceManager::resourceForId(RemoteMediaResourceIdentifier identifier)
{
    Locker locker { m_lock };
    return m_remoteMediaResources.get(identifier).get();
}

void RemoteMediaResourceManager::responseReceived(RemoteMediaResourceIdentifier identifier, const ResourceResponse& response, bool didPassAccessControlCheck, CompletionHandler<void(ShouldContinuePolicyCheck)>&& completionHandler)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->responseReceived(response, didPassAccessControlCheck, WTFMove(completionHandler));
    else
        completionHandler(ShouldContinuePolicyCheck::No);
}

void RemoteMediaResourceManager::redirectReceived(RemoteMediaResourceIdentifier identifier, ResourceRequest&& request, const ResourceResponse& response, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->redirectReceived(WTFMove(request), response, WTFMove(completionHandler));
    else
        completionHandler({ });
}

void RemoteMediaResourceManager::dataSent(RemoteMediaResourceIdentifier identifier, uint64_t bytesSent, uint64_t totalBytesToBeSent)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->dataSent(bytesSent, totalBytesToBeSent);
}

void RemoteMediaResourceManager::dataReceived(RemoteMediaResourceIdentifier identifier, IPC::SharedBufferReference&& buffer, CompletionHandler<void(std::optional<SharedMemory::Handle>&&)>&& completionHandler)
{
    auto resource = resourceForId(identifier);
    if (!resource)
        return completionHandler(std::nullopt);

    auto sharedMemory = buffer.sharedCopy();
    if (!sharedMemory)
        return completionHandler(std::nullopt);

    auto handle = sharedMemory->createHandle(SharedMemory::Protection::ReadOnly);
    if (!handle)
        return completionHandler(std::nullopt);

    resource->dataReceived(sharedMemory->createSharedBuffer(buffer.size()));
    completionHandler(WTFMove(handle));
}

void RemoteMediaResourceManager::accessControlCheckFailed(RemoteMediaResourceIdentifier identifier, const ResourceError& error)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->accessControlCheckFailed(error);
}

void RemoteMediaResourceManager::loadFailed(RemoteMediaResourceIdentifier identifier, const ResourceError& error)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->loadFailed(error);
}

void RemoteMediaResourceManager::loadFinished(RemoteMediaResourceIdentifier identifier, const NetworkLoadMetrics& metrics)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto resource = resourceForId(identifier))
        resource->loadFinished(metrics);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
