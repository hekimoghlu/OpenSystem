/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#include "NetscapePlugInStreamLoader.h"

#include "DocumentLoader.h"
#include "LoaderStrategy.h"
#include "PlatformStrategies.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>

#if ENABLE(CONTENT_EXTENSIONS)
#include "ResourceLoadInfo.h"
#endif

namespace WebCore {

// FIXME: Skip Content Security Policy check when associated plugin element is in a user agent shadow tree.
// See <https://bugs.webkit.org/show_bug.cgi?id=146663>.
NetscapePlugInStreamLoader::NetscapePlugInStreamLoader(LocalFrame& frame, NetscapePlugInStreamLoaderClient& client)
    : ResourceLoader(frame, ResourceLoaderOptions(
        SendCallbackPolicy::SendCallbacks,
        ContentSniffingPolicy::SniffContent,
        DataBufferingPolicy::DoNotBufferData,
        StoredCredentialsPolicy::Use,
        ClientCredentialPolicy::MayAskClientForCredentials,
        FetchOptions::Credentials::Include,
        SecurityCheckPolicy::SkipSecurityCheck,
        FetchOptions::Mode::NoCors,
        CertificateInfoPolicy::DoNotIncludeCertificateInfo,
        ContentSecurityPolicyImposition::DoPolicyCheck,
        DefersLoadingPolicy::AllowDefersLoading,
        CachingPolicy::AllowCaching))
    , m_client(client)
{
#if ENABLE(CONTENT_EXTENSIONS)
    m_resourceType = { ContentExtensions::ResourceType::Other };
#endif
}

NetscapePlugInStreamLoader::~NetscapePlugInStreamLoader() = default;

void NetscapePlugInStreamLoader::create(LocalFrame& frame, NetscapePlugInStreamLoaderClient& client, ResourceRequest&& request, CompletionHandler<void(RefPtr<NetscapePlugInStreamLoader>&&)>&& completionHandler)
{
    if (request.isNull())
        return completionHandler(nullptr);

    Ref loader = adoptRef(*new NetscapePlugInStreamLoader(frame, client));
    loader->init(WTFMove(request), [loader, completionHandler = WTFMove(completionHandler)] (bool initialized) mutable {
        if (!initialized)
            return completionHandler(nullptr);
        completionHandler(WTFMove(loader));
    });
}

bool NetscapePlugInStreamLoader::isDone() const
{
    return !m_client;
}

void NetscapePlugInStreamLoader::releaseResources()
{
    m_client = nullptr;
    ResourceLoader::releaseResources();
}

void NetscapePlugInStreamLoader::init(ResourceRequest&& request, CompletionHandler<void(bool)>&& completionHandler)
{
    ResourceLoader::init(WTFMove(request), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        if (!success)
            return completionHandler(false);
        ASSERT(!reachedTerminalState());
        protectedDocumentLoader()->addPlugInStreamLoader(*this);
        m_isInitialized = true;
        completionHandler(true);
    });
}

void NetscapePlugInStreamLoader::willSendRequest(ResourceRequest&& request, const ResourceResponse& redirectResponse, CompletionHandler<void(ResourceRequest&&)>&& callback)
{
    if (!m_client)
        return;

    m_client->willSendRequest(this, WTFMove(request), redirectResponse, [protectedThis = Ref { *this }, redirectResponse, callback = WTFMove(callback)] (ResourceRequest&& request) mutable {
        if (!request.isNull())
            protectedThis->willSendRequestInternal(WTFMove(request), redirectResponse, WTFMove(callback));
        else
            callback({ });
    });
}

void NetscapePlugInStreamLoader::didReceiveResponse(const ResourceResponse& response, CompletionHandler<void()>&& policyCompletionHandler)
{
    Ref<NetscapePlugInStreamLoader> protectedThis(*this);
    CompletionHandlerCallingScope completionHandlerCaller(WTFMove(policyCompletionHandler));

    if (m_client)
        m_client->didReceiveResponse(this, response);

    // Don't continue if the stream is cancelled
    if (!m_client)
        return;

    ResourceLoader::didReceiveResponse(response, [this, protectedThis = WTFMove(protectedThis), response, completionHandlerCaller = WTFMove(completionHandlerCaller)]() mutable {
        // Don't continue if the stream is cancelled
        if (!m_client)
            return;

        if (!response.isInHTTPFamily())
            return;

        if (m_client->wantsAllStreams())
            return;

        // Status code can be null when serving from a Web archive.
        if (response.httpStatusCode() && (response.httpStatusCode() < 100 || response.httpStatusCode() >= 400))
            cancel(platformStrategies()->loaderStrategy()->fileDoesNotExistError(response));
    });
}

void NetscapePlugInStreamLoader::didReceiveData(const SharedBuffer& buffer, long long encodedDataLength, DataPayloadType dataPayloadType)
{
    Ref protectedThis { *this };

    if (m_client)
        m_client->didReceiveData(this, buffer);

    ResourceLoader::didReceiveData(buffer, encodedDataLength, dataPayloadType);
}

void NetscapePlugInStreamLoader::didFinishLoading(const NetworkLoadMetrics& networkLoadMetrics)
{
    Ref protectedThis { *this };

    notifyDone();

    if (m_client)
        m_client->didFinishLoading(this);
    ResourceLoader::didFinishLoading(networkLoadMetrics);
}

void NetscapePlugInStreamLoader::didFail(const ResourceError& error)
{
    Ref protectedThis { *this };

    notifyDone();

    if (m_client)
        m_client->didFail(this, error);
    ResourceLoader::didFail(error);
}

void NetscapePlugInStreamLoader::willCancel(const ResourceError& error)
{
    if (m_client)
        m_client->didFail(this, error);
}

void NetscapePlugInStreamLoader::didCancel(LoadWillContinueInAnotherProcess)
{
    notifyDone();
}

void NetscapePlugInStreamLoader::notifyDone()
{
    if (!m_isInitialized)
        return;

    protectedDocumentLoader()->removePlugInStreamLoader(*this);
}


}
