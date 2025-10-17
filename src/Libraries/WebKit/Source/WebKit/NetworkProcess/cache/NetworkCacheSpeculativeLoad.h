/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION) || ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)

#include "NetworkCache.h"
#include "NetworkCacheEntry.h"
#include "NetworkLoadClient.h"
#include "PolicyDecision.h"
#include "PrivateRelayed.h"
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
enum class AdvancedPrivacyProtections : uint16_t;
}

namespace WebKit {

class NetworkLoad;

namespace NetworkCache {

class SpeculativeLoad final : public NetworkLoadClient {
    WTF_MAKE_TZONE_ALLOCATED(SpeculativeLoad);
public:
    using RevalidationCompletionHandler = CompletionHandler<void(std::unique_ptr<NetworkCache::Entry>)>;
    SpeculativeLoad(Cache&, const GlobalFrameID&, const WebCore::ResourceRequest&, std::unique_ptr<NetworkCache::Entry>, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, RevalidationCompletionHandler&&);

    virtual ~SpeculativeLoad();

    const WebCore::ResourceRequest& originalRequest() const { return m_originalRequest; }

    void cancel();

private:
    // NetworkLoadClient.
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) override { }
    bool isSynchronous() const override { return false; }
    bool isAllowedToAskUserForCredentials() const final { return false; }
    void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) override;
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) override;
    void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) override;
    void didFinishLoading(const WebCore::NetworkLoadMetrics&) override;
    void didFailLoading(const WebCore::ResourceError&) override;

    void didComplete();

    Ref<Cache> m_cache;
    RevalidationCompletionHandler m_completionHandler;
    WebCore::ResourceRequest m_originalRequest;

    RefPtr<NetworkLoad> m_networkLoad;

    WebCore::ResourceResponse m_response;

    WebCore::SharedBufferBuilder m_bufferedDataForCache;
    std::unique_ptr<NetworkCache::Entry> m_cacheEntry;
    bool m_didComplete { false };
    PrivateRelayed m_privateRelayed { PrivateRelayed::No };
};

bool requestsHeadersMatch(const WebCore::ResourceRequest& speculativeValidationRequest, const WebCore::ResourceRequest& actualRequest);

} // namespace NetworkCache
} // namespace WebKit

#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION) || ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
