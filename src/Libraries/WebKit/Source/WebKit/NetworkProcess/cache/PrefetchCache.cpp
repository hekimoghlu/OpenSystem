/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#include "PrefetchCache.h"

#include <WebCore/HTTPHeaderNames.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PrefetchCache);

PrefetchCache::Entry::Entry(WebCore::ResourceResponse&& response, PrivateRelayed privateRelayed, RefPtr<WebCore::FragmentedSharedBuffer>&& buffer)
    : response(WTFMove(response))
    , privateRelayed(privateRelayed)
    , buffer(WTFMove(buffer))
{
}

PrefetchCache::Entry::Entry(WebCore::ResourceResponse&& redirectResponse, WebCore::ResourceRequest&& redirectRequest)
    : response(WTFMove(redirectResponse)), redirectRequest(WTFMove(redirectRequest))
{
}

PrefetchCache::PrefetchCache()
    : m_expirationTimer(*this, &PrefetchCache::clearExpiredEntries)
{
}

PrefetchCache::~PrefetchCache()
{
}

void PrefetchCache::clear()
{
    m_expirationTimer.stop();
    m_sessionExpirationList.clear();
    if (m_sessionPrefetches)
        m_sessionPrefetches->clear();
}

std::unique_ptr<PrefetchCache::Entry> PrefetchCache::take(const URL& url)
{
    auto* resources = m_sessionPrefetches.get();
    if (!resources)
        return nullptr;
    m_sessionExpirationList.removeAllMatching([&url] (const auto& tuple) {
        return std::get<0>(tuple) == url;
    });
    auto entry = resources->take(url);
    ASSERT(!entry || !entry->response.httpHeaderField(WebCore::HTTPHeaderName::Vary).contains("Cookie"_s));
    return entry;
}

static const Seconds expirationTimeout { 5_s };

void PrefetchCache::store(const URL& requestURL, WebCore::ResourceResponse&& response, PrivateRelayed privateRelayed, RefPtr<WebCore::FragmentedSharedBuffer>&& buffer)
{
    if (!m_sessionPrefetches)
        m_sessionPrefetches = makeUnique<PrefetchEntriesMap>();
    auto addResult = m_sessionPrefetches->add(requestURL, makeUnique<PrefetchCache::Entry>(WTFMove(response), privateRelayed, WTFMove(buffer)));
    // Limit prefetches for same url to 1.
    if (!addResult.isNewEntry)
        return;
    m_sessionExpirationList.append(std::make_tuple(requestURL, WallTime::now()));
    if (!m_expirationTimer.isActive())
        m_expirationTimer.startOneShot(expirationTimeout);
}

void PrefetchCache::storeRedirect(const URL& requestUrl, WebCore::ResourceResponse&& redirectResponse, WebCore::ResourceRequest&& redirectRequest)
{
    if (!m_sessionPrefetches)
        m_sessionPrefetches = makeUnique<PrefetchEntriesMap>();
    redirectRequest.clearPurpose();
    m_sessionPrefetches->set(requestUrl, makeUnique<PrefetchCache::Entry>(WTFMove(redirectResponse), WTFMove(redirectRequest)));
    m_sessionExpirationList.append(std::make_tuple(requestUrl, WallTime::now()));
    if (!m_expirationTimer.isActive())
        m_expirationTimer.startOneShot(expirationTimeout);
}

void PrefetchCache::clearExpiredEntries()
{
    auto timeout = WallTime::now();
    while (!m_sessionExpirationList.isEmpty()) {
        auto& [requestUrl, timestamp] = m_sessionExpirationList.first();
        auto* resources = m_sessionPrefetches.get();
        ASSERT(resources);
        ASSERT(resources->contains(requestUrl));
        auto elapsed = timeout - timestamp;
        if (elapsed > expirationTimeout) {
            resources->remove(requestUrl);
            m_sessionExpirationList.removeFirst();
        } else {
            m_expirationTimer.startOneShot(expirationTimeout - elapsed);
            break;
        }
    }
}

}
