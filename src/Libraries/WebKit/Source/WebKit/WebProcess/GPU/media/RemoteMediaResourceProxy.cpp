/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#include "RemoteMediaResourceProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "RemoteMediaResourceManagerMessages.h"
#include "SharedBufferReference.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaResourceProxy);

RemoteMediaResourceProxy::RemoteMediaResourceProxy(Ref<IPC::Connection>&& connection, WebCore::PlatformMediaResource& platformMediaResource, RemoteMediaResourceIdentifier identifier)
    : m_connection(WTFMove(connection))
    , m_platformMediaResource(platformMediaResource)
    , m_id(identifier)
{
}

RemoteMediaResourceProxy::~RemoteMediaResourceProxy() = default;

Ref<WebCore::PlatformMediaResource> RemoteMediaResourceProxy::protectedMediaResource() const
{
    return m_platformMediaResource.get().releaseNonNull();
}

void RemoteMediaResourceProxy::responseReceived(WebCore::PlatformMediaResource&, const WebCore::ResourceResponse& response, CompletionHandler<void(WebCore::ShouldContinuePolicyCheck)>&& completionHandler)
{
    protectedConnection()->sendWithAsyncReply(Messages::RemoteMediaResourceManager::ResponseReceived(m_id, response, protectedMediaResource()->didPassAccessControlCheck()), [completionHandler = WTFMove(completionHandler)](auto shouldContinue) mutable {
        completionHandler(shouldContinue);
    });
}

void RemoteMediaResourceProxy::redirectReceived(WebCore::PlatformMediaResource&, WebCore::ResourceRequest&& request, const WebCore::ResourceResponse& response, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    protectedConnection()->sendWithAsyncReply(Messages::RemoteMediaResourceManager::RedirectReceived(m_id, request, response), [completionHandler = WTFMove(completionHandler)](WebCore::ResourceRequest&& request) mutable {
        completionHandler(WTFMove(request));
    });
}

bool RemoteMediaResourceProxy::shouldCacheResponse(WebCore::PlatformMediaResource&, const WebCore::ResourceResponse&)
{
    // TODO: need to check WebCoreNSURLSessionDataTaskClient::shouldCacheResponse()
    return false;
}

void RemoteMediaResourceProxy::dataSent(WebCore::PlatformMediaResource&, unsigned long long bytesSent, unsigned long long totalBytesToBeSent)
{
    protectedConnection()->send(Messages::RemoteMediaResourceManager::DataSent(m_id, bytesSent, totalBytesToBeSent), 0);
}

void RemoteMediaResourceProxy::dataReceived(WebCore::PlatformMediaResource&, const WebCore::SharedBuffer& buffer)
{
    protectedConnection()->sendWithAsyncReply(Messages::RemoteMediaResourceManager::DataReceived(m_id, IPC::SharedBufferReference { buffer }), [] (auto&& bufferHandle) {
        // Take ownership of shared memory and mark it as media-related memory.
        if (bufferHandle)
            bufferHandle->takeOwnershipOfMemory(WebCore::MemoryLedger::Media);
    }, 0);
}

void RemoteMediaResourceProxy::accessControlCheckFailed(WebCore::PlatformMediaResource&, const WebCore::ResourceError& error)
{
    protectedConnection()->send(Messages::RemoteMediaResourceManager::AccessControlCheckFailed(m_id, error), 0);
}

void RemoteMediaResourceProxy::loadFailed(WebCore::PlatformMediaResource&, const WebCore::ResourceError& error)
{
    protectedConnection()->send(Messages::RemoteMediaResourceManager::LoadFailed(m_id, error), 0);
}

void RemoteMediaResourceProxy::loadFinished(WebCore::PlatformMediaResource&, const WebCore::NetworkLoadMetrics& metrics)
{
    protectedConnection()->send(Messages::RemoteMediaResourceManager::LoadFinished(m_id, metrics), 0);
}

}

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
