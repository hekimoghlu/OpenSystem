/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "WebCoreAVCFResourceLoader.h"

#if ENABLE(VIDEO) && USE(AVFOUNDATION) && HAVE(AVFOUNDATION_LOADER_DELEGATE)

#include "CachedRawResource.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "NotImplemented.h"
#include "ResourceLoaderOptions.h"
#include "SharedBuffer.h"
#include <AVFoundationCF/AVFoundationCF.h>
#include <wtf/SoftLinking.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCoreAVCFResourceLoader);

Ref<WebCoreAVCFResourceLoader> WebCoreAVCFResourceLoader::create(MediaPlayerPrivateAVFoundationCF* parent, AVCFAssetResourceLoadingRequestRef avRequest)
{
    ASSERT(avRequest);
    ASSERT(parent);
    return adoptRef(*new WebCoreAVCFResourceLoader(parent, avRequest));
}

WebCoreAVCFResourceLoader::WebCoreAVCFResourceLoader(MediaPlayerPrivateAVFoundationCF* parent, AVCFAssetResourceLoadingRequestRef avRequest)
    : m_parent(parent)
    , m_avRequest(avRequest)
{
}

WebCoreAVCFResourceLoader::~WebCoreAVCFResourceLoader()
{
    stopLoading();
}

void WebCoreAVCFResourceLoader::startLoading()
{
    if (m_resource || !m_parent)
        return;

    RetainPtr<CFURLRequestRef> urlRequest = AVCFAssetResourceLoadingRequestGetURLRequest(m_avRequest.get());

    ResourceRequest resourceRequest(urlRequest.get());
    resourceRequest.setPriority(ResourceLoadPriority::Low);

    // ContentSecurityPolicyImposition::DoPolicyCheck is a placeholder value. It does not affect the request since Content Security Policy does not apply to raw resources.
    CachedResourceRequest request(WTFMove(resourceRequest), ResourceLoaderOptions(
        SendCallbackPolicy::SendCallbacks,
        ContentSniffingPolicy::DoNotSniffContent,
        DataBufferingPolicy::BufferData,
        StoredCredentialsPolicy::DoNotUse,
        ClientCredentialPolicy::CannotAskClientForCredentials,
        FetchOptions::Credentials::Omit,
        SecurityCheckPolicy::DoSecurityCheck,
        FetchOptions::Mode::NoCors,
        CertificateInfoPolicy::DoNotIncludeCertificateInfo,
        ContentSecurityPolicyImposition::DoPolicyCheck,
        DefersLoadingPolicy::AllowDefersLoading,
        CachingPolicy::DisallowCaching));

    CachedResourceLoader* loader = m_parent->player()->cachedResourceLoader();
    m_resource = loader ? loader->requestRawResource(WTFMove(request)).value_or(nullptr) : nullptr;
    if (m_resource)
        m_resource->addClient(*this);
    else {
        LOG_ERROR("Failed to start load for media at url %s", URL(CFURLRequestGetURL(urlRequest.get())).string().ascii().data());
        RetainPtr<CFErrorRef> error = adoptCF(CFErrorCreate(kCFAllocatorDefault, kCFErrorDomainCFNetwork, kCFURLErrorUnknown, nullptr));
        AVCFAssetResourceLoadingRequestFinishLoadingWithError(m_avRequest.get(), error.get());
    }
}

void WebCoreAVCFResourceLoader::stopLoading()
{
    if (!m_resource)
        return;

    m_resource->removeClient(*this);
    m_resource = 0;

    if (m_parent)
        m_parent->didStopLoadingRequest(m_avRequest.get());
}

void WebCoreAVCFResourceLoader::invalidate()
{
    if (!m_parent)
        return;

    m_parent = nullptr;

    callOnMainThread([protectedThis = Ref { *this }] () mutable {
        protectedThis->stopLoading();
    });
}

void WebCoreAVCFResourceLoader::responseReceived(CachedResource& resource, const ResourceResponse& response, CompletionHandler<void()>&& completionHandler)
{
    ASSERT_UNUSED(resource, &resource == m_resource);
    CompletionHandlerCallingScope completionHandlerCaller(WTFMove(completionHandler));

    int status = response.httpStatusCode();
    if (status && (status < 200 || status > 299)) {
        RetainPtr<CFErrorRef> error = adoptCF(CFErrorCreate(kCFAllocatorDefault, kCFErrorDomainCFNetwork, status, nullptr));
        AVCFAssetResourceLoadingRequestFinishLoadingWithError(m_avRequest.get(), error.get());
        return;
    }

    notImplemented();
}

void WebCoreAVCFResourceLoader::dataReceived(CachedResource& resource, const SharedBuffer&)
{
    fulfillRequestWithResource(resource);
}

void WebCoreAVCFResourceLoader::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&)
{
    if (resource.loadFailedOrCanceled()) {
        // <rdar://problem/13987417> Set the contentType of the contentInformationRequest to an empty
        // string to trigger AVAsset's playable value to complete loading.
        // FIXME: if ([m_avRequest.get() contentInformationRequest] && ![[m_avRequest.get() contentInformationRequest] contentType])
        // FIXME:    [[m_avRequest.get() contentInformationRequest] setContentType:@""];
        notImplemented();

        RetainPtr<CFErrorRef> error = adoptCF(CFErrorCreate(kCFAllocatorDefault, kCFErrorDomainCFNetwork, kCFURLErrorUnknown, nullptr));
        AVCFAssetResourceLoadingRequestFinishLoadingWithError(m_avRequest.get(), error.get());
    } else {
        fulfillRequestWithResource(resource);
        // FIXME: [m_avRequest.get() finishLoading];
        notImplemented();
    }
    stopLoading();
}

void WebCoreAVCFResourceLoader::fulfillRequestWithResource(CachedResource& resource)
{
    ASSERT_UNUSED(resource, &resource == m_resource);
    notImplemented();
}

}

#endif // ENABLE(VIDEO) && USE(AVFOUNDATION)
