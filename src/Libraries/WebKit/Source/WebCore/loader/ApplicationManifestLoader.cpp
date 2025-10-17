/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "ApplicationManifestLoader.h"

#if ENABLE(APPLICATION_MANIFEST)

#include "CachedApplicationManifest.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"

namespace WebCore {

ApplicationManifestLoader::ApplicationManifestLoader(DocumentLoader& documentLoader, const URL& url, bool useCredentials)
    : m_documentLoader(documentLoader)
    , m_url(url)
    , m_useCredentials(useCredentials)
{
}

ApplicationManifestLoader::~ApplicationManifestLoader()
{
    stopLoading();
}

bool ApplicationManifestLoader::startLoading()
{
    ASSERT(!m_resource);
    RefPtr frame = m_documentLoader->frame();
    if (!frame)
        return false;

    ResourceRequest resourceRequest = m_url;
    resourceRequest.setPriority(ResourceLoadPriority::Low);
#if !ERROR_DISABLED
    // Copy this because we may want to access it after transferring the
    // `resourceRequest` to the `request`. If we don't, then the LOG_ERROR
    // below won't print a URL.
    auto resourceRequestURL = resourceRequest.url();
#endif

    auto credentials = m_useCredentials ? FetchOptions::Credentials::Include : FetchOptions::Credentials::Omit;
    // The "linked resource fetch setup steps" are defined as part of:
    // https://html.spec.whatwg.org/#link-type-manifest
    auto options = ResourceLoaderOptions(
        SendCallbackPolicy::SendCallbacks,
        ContentSniffingPolicy::SniffContent,
        DataBufferingPolicy::BufferData,
        StoredCredentialsPolicy::DoNotUse,
        ClientCredentialPolicy::CannotAskClientForCredentials,
        credentials,
        SecurityCheckPolicy::DoSecurityCheck,
        FetchOptions::Mode::Cors,
        CertificateInfoPolicy::DoNotIncludeCertificateInfo,
        ContentSecurityPolicyImposition::DoPolicyCheck,
        DefersLoadingPolicy::AllowDefersLoading,
        CachingPolicy::AllowCaching);
    options.destination = FetchOptions::Destination::Manifest;
    options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;
    CachedResourceRequest request(WTFMove(resourceRequest), options);

    auto cachedResource = frame->document()->protectedCachedResourceLoader()->requestApplicationManifest(WTFMove(request));
    m_resource = cachedResource.value_or(nullptr);
    if (CachedResourceHandle resource = m_resource)
        resource->addClient(*this);
    else {
        LOG_ERROR("Failed to start load for application manifest at url %s (error: %s)", resourceRequestURL.string().ascii().data(), cachedResource.error().localizedDescription().utf8().data());
        return false;
    }

    return true;
}

void ApplicationManifestLoader::stopLoading()
{
    if (CachedResourceHandle resource = std::exchange(m_resource, nullptr))
        resource->removeClient(*this);
}

std::optional<ApplicationManifest>& ApplicationManifestLoader::processManifest()
{
    if (!m_processedManifest) {
        if (CachedResourceHandle resource = m_resource) {
            auto manifestURL = m_url;
            auto documentURL = m_documentLoader->url();
            auto frame = m_documentLoader->frame();
            RefPtr document = frame ? frame->document() : nullptr;
            m_processedManifest = resource->process(manifestURL, documentURL, document.get());
        }
    }
    return m_processedManifest;
}

void ApplicationManifestLoader::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    ASSERT_UNUSED(resource, &resource == m_resource);
    Ref { m_documentLoader.get() }->finishedLoadingApplicationManifest(*this);
}

} // namespace WebCore

#endif // ENABLE(APPLICATION_MANIFEST)
