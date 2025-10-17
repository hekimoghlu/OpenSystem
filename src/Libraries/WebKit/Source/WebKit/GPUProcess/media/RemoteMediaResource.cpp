/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include "RemoteMediaResource.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "RemoteMediaPlayerProxy.h"
#include "RemoteMediaResourceLoader.h"
#include "RemoteMediaResourceManager.h"
#include <WebCore/ResourceResponse.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaResource);

using namespace WebCore;

Ref<RemoteMediaResource> RemoteMediaResource::create(RemoteMediaResourceManager& remoteMediaResourceManager, RemoteMediaPlayerProxy& remoteMediaPlayerProxy, RemoteMediaResourceIdentifier identifier)
{
    return adoptRef(*new RemoteMediaResource(remoteMediaResourceManager, remoteMediaPlayerProxy, identifier));
}

RemoteMediaResource::RemoteMediaResource(RemoteMediaResourceManager& remoteMediaResourceManager, RemoteMediaPlayerProxy& remoteMediaPlayerProxy, RemoteMediaResourceIdentifier identifier)
    : m_remoteMediaResourceManager(remoteMediaResourceManager)
    , m_remoteMediaPlayerProxy(remoteMediaPlayerProxy)
    , m_id(identifier)
{
    ASSERT(isMainRunLoop());
}

RemoteMediaResource::~RemoteMediaResource()
{
    ASSERT(m_shutdown);
}

void RemoteMediaResource::shutdown()
{
    if (m_shutdown.exchange(true))
        return;

    setClient(nullptr);

    if (RefPtr remoteMediaResourceManager = m_remoteMediaResourceManager.get())
        remoteMediaResourceManager->removeMediaResource(m_id);

    ensureOnMainRunLoop([remoteMediaPlayerProxy = WTFMove(m_remoteMediaPlayerProxy), id = m_id] {
        if (remoteMediaPlayerProxy)
            remoteMediaPlayerProxy->removeResource(id);
    });
}

bool RemoteMediaResource::didPassAccessControlCheck() const
{
    return m_didPassAccessControlCheck;
}

void RemoteMediaResource::responseReceived(const ResourceResponse& response, bool didPassAccessControlCheck, CompletionHandler<void(ShouldContinuePolicyCheck)>&& completionHandler)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    auto client = this->client();
    if (!client)
        return completionHandler(ShouldContinuePolicyCheck::No);

    m_didPassAccessControlCheck = didPassAccessControlCheck;
    client->responseReceived(*this, response, [protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)](auto shouldContinue) mutable {
        if (shouldContinue == ShouldContinuePolicyCheck::No) {
            ensureOnMainThread([protectedThis] {
                protectedThis->shutdown();
            });
        }

        completionHandler(shouldContinue);
    });
}

void RemoteMediaResource::redirectReceived(ResourceRequest&& request, const ResourceResponse& response, CompletionHandler<void(ResourceRequest&&)>&& completionHandler)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto client = this->client())
        client->redirectReceived(*this, WTFMove(request), response, WTFMove(completionHandler));
}

void RemoteMediaResource::dataSent(uint64_t bytesSent, uint64_t totalBytesToBeSent)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto client = this->client())
        client->dataSent(*this, bytesSent, totalBytesToBeSent);
}

void RemoteMediaResource::dataReceived(const SharedBuffer& data)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto client = this->client())
        client->dataReceived(*this, data);
}

void RemoteMediaResource::accessControlCheckFailed(const ResourceError& error)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    m_didPassAccessControlCheck = false;
    if (auto client = this->client())
        client->accessControlCheckFailed(*this, error);
}

void RemoteMediaResource::loadFailed(const ResourceError& error)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto client = this->client())
        client->loadFailed(*this, error);
}

void RemoteMediaResource::loadFinished(const NetworkLoadMetrics& metrics)
{
    assertIsCurrent(RemoteMediaResourceLoader::defaultQueue());

    if (auto client = this->client())
        client->loadFinished(*this, metrics);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
