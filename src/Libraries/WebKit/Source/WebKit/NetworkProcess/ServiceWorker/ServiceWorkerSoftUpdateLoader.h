/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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

#include "NetworkCacheEntry.h"
#include "NetworkLoadClient.h"
#include <WebCore/ContentSecurityPolicyResponseHeaders.h>
#include <WebCore/CrossOriginEmbedderPolicy.h>
#include <WebCore/FetchOptions.h>
#include <WebCore/ServiceWorkerJobData.h>
#include <wtf/CheckedPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct ServiceWorkerJobData;
struct WorkerFetchResult;
class TextResourceDecoder;
}

namespace WebKit {

class NetworkLoad;
class NetworkSession;

class ServiceWorkerSoftUpdateLoader final : public NetworkLoadClient, public CanMakeWeakPtr<ServiceWorkerSoftUpdateLoader>, public CanMakeCheckedPtr<ServiceWorkerSoftUpdateLoader> {
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerSoftUpdateLoader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ServiceWorkerSoftUpdateLoader);
public:
    using Handler = CompletionHandler<void(WebCore::WorkerFetchResult&&)>;
    ServiceWorkerSoftUpdateLoader(NetworkSession&, WebCore::ServiceWorkerJobData&&, bool shouldRefreshCache, WebCore::ResourceRequest&&, Handler&&);
    ~ServiceWorkerSoftUpdateLoader();
    
private:
    // NetworkLoadClient.
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) final { }
    bool isSynchronous() const final { return false; }
    bool isAllowedToAskUserForCredentials() const final { return false; }
    void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) final;
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) final;
    void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) final;
    void didFinishLoading(const WebCore::NetworkLoadMetrics&) final;
    void didFailLoading(const WebCore::ResourceError&) final;

    void loadWithCacheEntry(NetworkCache::Entry&);
    void loadFromNetwork(NetworkSession&, WebCore::ResourceRequest&&);
    void fail(WebCore::ResourceError&&);
    void didComplete();
    WebCore::ResourceError processResponse(const WebCore::ResourceResponse&);

    Handler m_completionHandler;
    WebCore::ServiceWorkerJobData m_jobData;
    RefPtr<NetworkLoad> m_networkLoad;
    WeakPtr<NetworkSession> m_session;

    String m_responseEncoding;
    String m_referrerPolicy;
    WebCore::ContentSecurityPolicyResponseHeaders m_contentSecurityPolicy;
    WebCore::CrossOriginEmbedderPolicy m_crossOriginEmbedderPolicy;

    std::unique_ptr<NetworkCache::Entry> m_cacheEntry;
    RefPtr<WebCore::TextResourceDecoder> m_decoder;
    StringBuilder m_script;
    WebCore::CertificateInfo m_certificateInfo;
};

} // namespace WebKit
