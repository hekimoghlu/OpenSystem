/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#pragma once

#include "CachedResourceLoader.h"
#include "FrameLoaderTypes.h"
#include "ResourceLoader.h"
#include <wtf/CompletionHandler.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>
 
namespace WebCore {

class CachedResource;
class CachedResourceLoader;
class NetworkLoadMetrics;
class ResourceRequest;
class SecurityOrigin;

class SubresourceLoader final : public ResourceLoader {
public:
    WEBCORE_EXPORT static void create(LocalFrame&, CachedResource&, ResourceRequest&&, const ResourceLoaderOptions&, CompletionHandler<void(RefPtr<SubresourceLoader>&&)>&&);

    virtual ~SubresourceLoader();

    void cancelIfNotFinishing();
    bool isSubresourceLoader() const final;
    CachedResource* cachedResource() const final { return m_resource.get(); };
    CachedResourceHandle<CachedResource> protectedCachedResource() const { return cachedResource(); }

    WEBCORE_EXPORT const HTTPHeaderMap* originalHeaders() const;

    const SecurityOrigin* origin() const { return m_origin.get(); }
    SecurityOrigin* origin() { return m_origin.get(); }
    RefPtr<SecurityOrigin> protectedOrigin() const;
#if PLATFORM(IOS_FAMILY)
    void startLoading() final;

    // FIXME: What is an "iOS" original request? Why is it necessary?
    const ResourceRequest& iOSOriginalRequest() const final { return m_iOSOriginalRequest; }
#endif

    unsigned redirectCount() const { return m_redirectCount; }

    void markInAsyncResponsePolicyCheck() { m_inAsyncResponsePolicyCheck = true; }
    void didReceiveResponsePolicy();

    void clearRequestCountTracker() { m_requestCountTracker = std::nullopt; }
    void resetRequestCountTracker(CachedResourceLoader& loader, const CachedResource& resource) { m_requestCountTracker = RequestCountTracker { loader, resource }; }

private:
    SubresourceLoader(LocalFrame&, CachedResource&, const ResourceLoaderOptions&);

    void init(ResourceRequest&&, CompletionHandler<void(bool)>&&) final;

    void willSendRequestInternal(ResourceRequest&&, const ResourceResponse& redirectResponse, CompletionHandler<void(ResourceRequest&&)>&&) final;
    void didSendData(unsigned long long bytesSent, unsigned long long totalBytesToBeSent) final;
    void didReceiveResponse(const ResourceResponse&, CompletionHandler<void()>&& policyCompletionHandler) final;
    void didReceiveBuffer(const FragmentedSharedBuffer&, long long encodedDataLength, DataPayloadType) final;
    void didFinishLoading(const NetworkLoadMetrics&) final;
    void didFail(const ResourceError&) final;
    void willCancel(const ResourceError&) final;
    void didCancel(LoadWillContinueInAnotherProcess) final;
    
    void updateReferrerPolicy(const String&);

#if PLATFORM(COCOA)
    void willCacheResponseAsync(ResourceHandle*, NSCachedURLResponse*, CompletionHandler<void(NSCachedURLResponse *)>&&) final;
#endif

    void releaseResources() final;

    bool responseHasHTTPStatusCodeError() const;
    Expected<void, String> checkResponseCrossOriginAccessControl(const ResourceResponse&);
    Expected<void, String> checkRedirectionCrossOriginAccessControl(const ResourceRequest& previousRequest, const ResourceResponse&, ResourceRequest& newRequest);

    void didReceiveDataOrBuffer(const FragmentedSharedBuffer&, long long encodedDataLength, DataPayloadType);

    void notifyDone(LoadCompletionType);

    void reportResourceTiming(const NetworkLoadMetrics&);

#if USE(QUICK_LOOK)
    bool shouldCreatePreviewLoaderForResponse(const ResourceResponse&) const;
    void didReceivePreviewResponse(const ResourceResponse&) final;
#endif

    enum SubresourceLoaderState {
        Uninitialized,
        Initialized,
        Finishing,
#if PLATFORM(IOS_FAMILY)
        CancelledWhileInitializing
#endif
    };

    class RequestCountTracker {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    public:
        RequestCountTracker(CachedResourceLoader&, const CachedResource&);
        RequestCountTracker(RequestCountTracker&&);
        RequestCountTracker& operator=(RequestCountTracker&&);
        ~RequestCountTracker();
    private:
        WeakPtr<CachedResourceLoader> m_cachedResourceLoader;
        WeakPtr<CachedResource> m_resource;
    };

#if PLATFORM(IOS_FAMILY)
    ResourceRequest m_iOSOriginalRequest;
#endif
    WeakPtr<CachedResource> m_resource;
    SubresourceLoaderState m_state;
    std::optional<RequestCountTracker> m_requestCountTracker;
    RefPtr<SecurityOrigin> m_origin;
    CompletionHandler<void()> m_policyForResponseCompletionHandler;
    ResourceResponse m_previousPartResponse;
    unsigned m_redirectCount { 0 };
    bool m_loadingMultipartContent { false };
    bool m_inAsyncResponsePolicyCheck { false };
    FetchMetadataSite m_site { FetchMetadataSite::SameOrigin };
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SubresourceLoader)
static bool isType(const WebCore::ResourceLoader& loader) { return loader.isSubresourceLoader(); }
SPECIALIZE_TYPE_TRAITS_END()
