/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "AsyncRevalidation.h"

#if ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
#include <WebCore/CacheValidation.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
namespace NetworkCache {

static inline WebCore::ResourceRequest constructRevalidationRequest(const Key& key, const WebCore::ResourceRequest& request, const Entry& entry)
{
    WebCore::ResourceRequest revalidationRequest = request;
    if (!key.partition().isEmpty())
        revalidationRequest.setCachePartition(key.partition());
    ASSERT_WITH_MESSAGE(key.range().isEmpty(), "range is not supported");

    revalidationRequest.makeUnconditional();
    auto eTag = entry.response().httpHeaderField(WebCore::HTTPHeaderName::ETag);
    if (!eTag.isEmpty())
        revalidationRequest.setHTTPHeaderField(WebCore::HTTPHeaderName::IfNoneMatch, eTag);

    auto lastModified = entry.response().httpHeaderField(WebCore::HTTPHeaderName::LastModified);
    if (!lastModified.isEmpty())
        revalidationRequest.setHTTPHeaderField(WebCore::HTTPHeaderName::IfModifiedSince, lastModified);

    revalidationRequest.setPriority(WebCore::ResourceLoadPriority::Low);

    return revalidationRequest;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(AsyncRevalidation);

void AsyncRevalidation::cancel()
{
    if (m_load)
        m_load->cancel();
}

void AsyncRevalidation::staleWhileRevalidateEnding()
{
    if (m_completionHandler)
        m_completionHandler(Result::Timeout);
}

Ref<AsyncRevalidation> AsyncRevalidation::create(Cache& cache, const GlobalFrameID& frameID, const WebCore::ResourceRequest& request, std::unique_ptr<NetworkCache::Entry>&& entry, std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections, CompletionHandler<void(Result)>&& handler)
{
    return adoptRef(*new AsyncRevalidation(cache, frameID, request, WTFMove(entry), isNavigatingToAppBoundDomain, allowPrivacyProxy, advancedPrivacyProtections, WTFMove(handler)));
}

AsyncRevalidation::AsyncRevalidation(Cache& cache, const GlobalFrameID& frameID, const WebCore::ResourceRequest& request, std::unique_ptr<NetworkCache::Entry>&& entry, std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections, CompletionHandler<void(Result)>&& handler)
    : m_timer(*this, &AsyncRevalidation::staleWhileRevalidateEnding)
    , m_completionHandler(WTFMove(handler))
{
    auto key = entry->key();
    auto revalidationRequest = constructRevalidationRequest(key, request, *entry.get());
    auto age = WebCore::computeCurrentAge(entry->response(), entry->timeStamp());
    auto lifetime = WebCore::computeFreshnessLifetimeForHTTPFamily(entry->response(), entry->timeStamp());
    auto responseMaxStaleness = entry->response().cacheControlStaleWhileRevalidate();
    ASSERT(responseMaxStaleness);
    m_timer.startOneShot(*responseMaxStaleness + (lifetime - age));
    m_load = makeUnique<SpeculativeLoad>(cache, frameID, WTFMove(revalidationRequest), WTFMove(entry), isNavigatingToAppBoundDomain, allowPrivacyProxy, advancedPrivacyProtections, [this, key, revalidationRequest](auto&& revalidatedEntry) {
        ASSERT(!revalidatedEntry || !revalidatedEntry->needsValidation());
        ASSERT(!revalidatedEntry || revalidatedEntry->key() == key);
        if (m_completionHandler)
            m_completionHandler(revalidatedEntry ? Result::Success : Result::Failure);
    });
}

} // namespace NetworkCache
} // namespace WebKit

#endif // ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
