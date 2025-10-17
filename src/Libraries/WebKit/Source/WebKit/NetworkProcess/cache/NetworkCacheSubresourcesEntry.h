/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)

#include "NetworkCacheStorage.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>

namespace WebKit::NetworkCache {

class SubresourceInfo {
    WTF_MAKE_TZONE_ALLOCATED(SubresourceInfo);
public:
    SubresourceInfo(Key&& key, WallTime lastSeen, WallTime firstSeen)
        : m_key(WTFMove(key))
        , m_lastSeen(lastSeen)
        , m_firstSeen(firstSeen)
        , m_isTransient(true) { }
    SubresourceInfo(Key&& key, WallTime lastSeen, WallTime firstSeen, bool isSameSite, bool isAppInitiated, URL&& firstPartyForCookies, WebCore::HTTPHeaderMap&& requestHeaders, WebCore::ResourceLoadPriority priority)
        : m_key(WTFMove(key))
        , m_lastSeen(lastSeen)
        , m_firstSeen(firstSeen)
        , m_isTransient(false)
        , m_isSameSite(isSameSite)
        , m_isAppInitiated(isAppInitiated)
        , m_firstPartyForCookies(WTFMove(firstPartyForCookies))
        , m_requestHeaders(WTFMove(requestHeaders))
        , m_priority(priority) { }
    SubresourceInfo(const Key&, const WebCore::ResourceRequest&, const SubresourceInfo* previousInfo);

    const Key& key() const { return m_key; }
    WallTime lastSeen() const { return m_lastSeen; }
    WallTime firstSeen() const { return m_firstSeen; }

    bool isTransient() const { return m_isTransient; }
    const URL& firstPartyForCookies() const { ASSERT(!m_isTransient); return m_firstPartyForCookies; }
    const WebCore::HTTPHeaderMap& requestHeaders() const { ASSERT(!m_isTransient); return m_requestHeaders; }
    WebCore::ResourceLoadPriority priority() const { ASSERT(!m_isTransient); return m_priority; }

    bool isSameSite() const { ASSERT(!m_isTransient); return m_isSameSite; }
    bool isTopSite() const { return false; }

    void setNonTransient() { m_isTransient = false; }

    bool isFirstParty() const;

    bool isAppInitiated() const { return m_isAppInitiated; }
    void setIsAppInitiated(bool isAppInitiated) { m_isAppInitiated = isAppInitiated; }

private:
    Key m_key;
    WallTime m_lastSeen;
    WallTime m_firstSeen;
    bool m_isTransient { false };
    bool m_isSameSite { false };
    bool m_isAppInitiated { true };
    URL m_firstPartyForCookies;
    WebCore::HTTPHeaderMap m_requestHeaders;
    WebCore::ResourceLoadPriority m_priority;
};

struct SubresourceLoad {
    WTF_MAKE_TZONE_ALLOCATED(SubresourceLoad);
    WTF_MAKE_NONCOPYABLE(SubresourceLoad);
public:
    SubresourceLoad(const WebCore::ResourceRequest& request, const Key& key)
        : request(request)
        , key(key)
    { }

    WebCore::ResourceRequest request;
    Key key;
};

class SubresourcesEntry {
    WTF_MAKE_TZONE_ALLOCATED(SubresourcesEntry);
    WTF_MAKE_NONCOPYABLE(SubresourcesEntry);
public:
    SubresourcesEntry(Key&&, const Vector<std::unique_ptr<SubresourceLoad>>&);
    explicit SubresourcesEntry(const Storage::Record&);

    Storage::Record encodeAsStorageRecord() const;
    static std::unique_ptr<SubresourcesEntry> decodeStorageRecord(const Storage::Record&);

    const Key& key() const { return m_key; }
    WallTime timeStamp() const { return m_timeStamp; }
    Vector<SubresourceInfo>& subresources() { return m_subresources; }

    void updateSubresourceLoads(const Vector<std::unique_ptr<SubresourceLoad>>&);

private:
    Key m_key;
    WallTime m_timeStamp;
    Vector<SubresourceInfo> m_subresources;
};

} // namespace WebKit::NetworkCache

#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)
