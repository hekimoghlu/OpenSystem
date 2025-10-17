/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include "ApplicationCacheResourceLoader.h"

#include "CachedResourceLoader.h"

namespace WebCore {

RefPtr<ApplicationCacheResourceLoader> ApplicationCacheResourceLoader::create(unsigned type, CachedResourceLoader& loader, ResourceRequest&& request, CompletionHandler<void(ResourceOrError&&)>&& callback)
{
    ResourceLoaderOptions options;
    options.storedCredentialsPolicy = StoredCredentialsPolicy::Use;
    options.credentials = FetchOptions::Credentials::Include;
    options.applicationCacheMode = ApplicationCacheMode::Bypass;
    options.certificateInfoPolicy = CertificateInfoPolicy::IncludeCertificateInfo;
    CachedResourceRequest cachedResourceRequest { WTFMove(request), options };
    auto resource = loader.requestRawResource(WTFMove(cachedResourceRequest));
    if (!resource.has_value()) {
        callback(makeUnexpected(Error::CannotCreateResource));
        return nullptr;
    }
    return adoptRef(*new ApplicationCacheResourceLoader { type, WTFMove(resource.value()), WTFMove(callback) });
}

ApplicationCacheResourceLoader::ApplicationCacheResourceLoader(unsigned type, CachedResourceHandle<CachedRawResource>&& resource, CompletionHandler<void(ResourceOrError&&)>&& callback)
    : m_type(type)
    , m_resource(WTFMove(resource))
    , m_callback(WTFMove(callback))
{
    m_resource->addClient(*this);
}

ApplicationCacheResourceLoader::~ApplicationCacheResourceLoader()
{
    if (auto callback = WTFMove(m_callback))
        callback(makeUnexpected(Error::Abort));

    if (m_resource)
        m_resource->removeClient(*this);
}

void ApplicationCacheResourceLoader::cancel(Error error)
{
    Ref protectedThis { *this };

    if (auto callback = WTFMove(m_callback))
        callback(makeUnexpected(error));

    if (m_resource) {
        m_resource->removeClient(*this);
        m_resource = nullptr;
    }
}

void ApplicationCacheResourceLoader::responseReceived(CachedResource& resource, const ResourceResponse& response, CompletionHandler<void()>&& completionHandler)
{
    ASSERT_UNUSED(resource, &resource == m_resource);
    CompletionHandlerCallingScope completionHandlerCaller(WTFMove(completionHandler));

    if (response.httpStatusCode() == 404 || response.httpStatusCode() == 410) {
        cancel(Error::NotFound);
        return;
    }

    if (response.httpStatusCode() == 304) {
        notifyFinished(*m_resource, { }, LoadWillContinueInAnotherProcess::No);
        return;
    }

    if (response.httpStatusCode() / 100 != 2) {
        cancel(Error::NotOK);
        return;
    }

    m_applicationCacheResource = ApplicationCacheResource::create(m_resource->url(), response, m_type);
}

void ApplicationCacheResourceLoader::dataReceived(CachedResource&, const SharedBuffer& buffer)
{
    m_applicationCacheResource->append(buffer);
}

void ApplicationCacheResourceLoader::redirectReceived(CachedResource&, ResourceRequest&& newRequest, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&& callback)
{
    m_hasRedirection = true;
    bool isRedirectionDisallowed = (m_type & ApplicationCacheResource::Type::Manifest) || (m_type & ApplicationCacheResource::Explicit) || (m_type & ApplicationCacheResource::Fallback);

    if (isRedirectionDisallowed) {
        cancel(Error::RedirectForbidden);
        callback({ });
        return;
    }
    callback(WTFMove(newRequest));
}

void ApplicationCacheResourceLoader::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    Ref protectedThis { *this };

    ASSERT_UNUSED(resource, &resource == m_resource);

    if (m_resource->errorOccurred()) {
        cancel(Error::NetworkError);
        return;
    }
    if (auto callback = WTFMove(m_callback))
        callback(WTFMove(m_applicationCacheResource));

    CachedResourceHandle<CachedRawResource> resourceHandle;
    std::swap(resourceHandle, m_resource);
    if (resourceHandle)
        resourceHandle->removeClient(*this);
}

} // namespace WebCore
