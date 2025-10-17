/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION) || ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
#include "NetworkCacheSpeculativeLoad.h"

#include "Logging.h"
#include "NetworkCache.h"
#include "NetworkLoad.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include <WebCore/HTTPStatusCodes.h>
#include <WebCore/NetworkStorageSession.h>
#include <pal/SessionID.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
namespace NetworkCache {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(SpeculativeLoad);

SpeculativeLoad::SpeculativeLoad(Cache& cache, const GlobalFrameID& globalFrameID, const ResourceRequest& request, std::unique_ptr<NetworkCache::Entry> cacheEntryForValidation, std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain, bool allowPrivacyProxy, OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections, RevalidationCompletionHandler&& completionHandler)
    : m_cache(cache)
    , m_completionHandler(WTFMove(completionHandler))
    , m_originalRequest(request)
    , m_bufferedDataForCache(FragmentedSharedBuffer::create())
    , m_cacheEntry(WTFMove(cacheEntryForValidation))
{
    ASSERT(!m_cacheEntry || m_cacheEntry->needsValidation());

    auto* networkSession = m_cache->networkProcess().networkSession(m_cache->sessionID());
    if (!networkSession) {
        RunLoop::main().dispatch([completionHandler = WTFMove(m_completionHandler)]() mutable {
            completionHandler(nullptr);
        });
        return;
    }

    NetworkLoadParameters parameters;
    parameters.webPageProxyID = globalFrameID.webPageProxyID;
    parameters.webPageID = globalFrameID.webPageID;
    parameters.webFrameID = globalFrameID.frameID;
    parameters.storedCredentialsPolicy = StoredCredentialsPolicy::Use;
    parameters.contentSniffingPolicy = ContentSniffingPolicy::DoNotSniffContent;
    parameters.contentEncodingSniffingPolicy = ContentEncodingSniffingPolicy::Default;
    parameters.request = m_originalRequest;
    parameters.isNavigatingToAppBoundDomain = isNavigatingToAppBoundDomain;
    parameters.allowPrivacyProxy = allowPrivacyProxy;
    parameters.advancedPrivacyProtections = advancedPrivacyProtections;
    m_networkLoad = NetworkLoad::create(*this, WTFMove(parameters), *networkSession);
    m_networkLoad->startWithScheduling();
}

SpeculativeLoad::~SpeculativeLoad()
{
    ASSERT(!m_networkLoad);
}

void SpeculativeLoad::cancel()
{
    if (!m_networkLoad)
        return;
    m_networkLoad->cancel();
    m_networkLoad = nullptr;
    m_completionHandler(nullptr);
}

void SpeculativeLoad::willSendRedirectedRequest(ResourceRequest&& request, ResourceRequest&& redirectRequest, ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    LOG(NetworkCacheSpeculativePreloading, "Speculative redirect %s -> %s", request.url().string().utf8().data(), redirectRequest.url().string().utf8().data());

    std::optional<Seconds> maxAgeCap;
    if (auto* networkStorageSession = m_cache->networkProcess().storageSession(m_cache->sessionID()))
        maxAgeCap = networkStorageSession->maxAgeCacheCap(request);
    m_cacheEntry = m_cache->storeRedirect(request, redirectResponse, redirectRequest, maxAgeCap);
    // Create a synthetic cache entry if we can't store.
    if (!m_cacheEntry)
        m_cacheEntry = m_cache->makeRedirectEntry(request, redirectResponse, redirectRequest);

    // Don't follow the redirect. The redirect target will be registered for speculative load when it is loaded.
    didComplete();
    completionHandler({ });
}

void SpeculativeLoad::didReceiveResponse(ResourceResponse&& receivedResponse, PrivateRelayed privateRelayed, ResponseCompletionHandler&& completionHandler)
{
    m_response = receivedResponse;
    m_privateRelayed = privateRelayed;

    if (m_response.isMultipart())
        m_bufferedDataForCache.reset();

    bool validationSucceeded = m_response.httpStatusCode() == httpStatus304NotModified;
    if (validationSucceeded && m_cacheEntry)
        m_cacheEntry = m_cache->update(m_originalRequest, *m_cacheEntry, m_response, privateRelayed);
    else
        m_cacheEntry = nullptr;

    completionHandler(PolicyAction::Use);
}

void SpeculativeLoad::didReceiveBuffer(const WebCore::FragmentedSharedBuffer& buffer, uint64_t reportedEncodedDataLength)
{
    ASSERT(!m_cacheEntry);

    if (m_bufferedDataForCache) {
        // Prevent memory growth in case of streaming data.
        const size_t maximumCacheBufferSize = 10 * 1024 * 1024;
        if (m_bufferedDataForCache.size() + buffer.size() <= maximumCacheBufferSize)
            m_bufferedDataForCache.append(buffer);
        else
            m_bufferedDataForCache.reset();
    }
}

void SpeculativeLoad::didFinishLoading(const WebCore::NetworkLoadMetrics&)
{
    if (m_didComplete)
        return;
    if (!m_cacheEntry && m_bufferedDataForCache) {
        m_cacheEntry = m_cache->store(m_originalRequest, m_response, m_privateRelayed, m_bufferedDataForCache.get(), [](auto&& mappedBody) { });
        // Create a synthetic cache entry if we can't store.
        if (!m_cacheEntry && isStatusCodeCacheableByDefault(m_response.httpStatusCode()))
            m_cacheEntry = m_cache->makeEntry(m_originalRequest, m_response, m_privateRelayed, m_bufferedDataForCache.take());
    }

    didComplete();
}

void SpeculativeLoad::didFailLoading(const ResourceError&)
{
    if (m_didComplete)
        return;
    m_cacheEntry = nullptr;

    didComplete();
}

void SpeculativeLoad::didComplete()
{
    RELEASE_ASSERT(RunLoop::isMain());

    if (m_didComplete)
        return;
    m_didComplete = true;
    m_networkLoad = nullptr;

    // Make sure speculatively revalidated resources do not get validated by the NetworkResourceLoader again.
    if (m_cacheEntry)
        m_cacheEntry->setNeedsValidation(false);

    m_completionHandler(WTFMove(m_cacheEntry));
}

#if !LOG_DISABLED

static void dumpHTTPHeadersDiff(const HTTPHeaderMap& headersA, const HTTPHeaderMap& headersB)
{
    auto aEnd = headersA.end();
    for (auto it = headersA.begin(); it != aEnd; ++it) {
        String valueB = headersB.get(it->key);
        if (valueB.isNull())
            LOG(NetworkCacheSpeculativePreloading, "* '%s' HTTP header is only in first request (value: %s)", it->key.utf8().data(), it->value.utf8().data());
        else if (it->value != valueB)
            LOG(NetworkCacheSpeculativePreloading, "* '%s' HTTP header differs in both requests: %s != %s", it->key.utf8().data(), it->value.utf8().data(), valueB.utf8().data());
    }
    auto bEnd = headersB.end();
    for (auto it = headersB.begin(); it != bEnd; ++it) {
        if (!headersA.contains(it->key))
            LOG(NetworkCacheSpeculativePreloading, "* '%s' HTTP header is only in second request (value: %s)", it->key.utf8().data(), it->value.utf8().data());
    }
}

#endif

bool requestsHeadersMatch(const ResourceRequest& speculativeValidationRequest, const ResourceRequest& actualRequest)
{
    ASSERT(!actualRequest.isConditional());
    ResourceRequest speculativeRequest = speculativeValidationRequest;
    speculativeRequest.makeUnconditional();

    if (speculativeRequest.httpHeaderFields() != actualRequest.httpHeaderFields()) {
        LOG(NetworkCacheSpeculativePreloading, "Cannot reuse speculatively validated entry because HTTP headers used for validation do not match");
#if !LOG_DISABLED
        dumpHTTPHeadersDiff(speculativeRequest.httpHeaderFields(), actualRequest.httpHeaderFields());
#endif
        return false;
    }
    return true;
}

} // namespace NetworkCache
} // namespace WebKit

#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION) || ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
