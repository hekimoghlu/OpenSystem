/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include "DNSCache.h"

#include <wtf/glib/RunLoopSourcePriority.h>

namespace WebKit {

static const Seconds expireInterval = 60_s;
static const unsigned maxCacheSize = 400;

Ref<DNSCache> DNSCache::create()
{
    return adoptRef(*new DNSCache);
}

DNSCache::DNSCache()
    : m_expiredTimer(RunLoop::main(), this, &DNSCache::removeExpiredResponsesFired)
{
    m_expiredTimer.setPriority(RunLoopSourcePriority::ReleaseUnusedResourcesTimer);
}

DNSCache::DNSCacheMap& DNSCache::mapForType(Type type)
{
    switch (type) {
    case Type::Default:
        return m_dnsMap;
    case Type::IPv4Only:
        return m_ipv4Map;
    case Type::IPv6Only:
        return m_ipv6Map;
    }

    RELEASE_ASSERT_NOT_REACHED();
    return m_dnsMap;
}

std::optional<Vector<GRefPtr<GInetAddress>>> DNSCache::lookup(const CString& host, Type type)
{
    Locker locker { m_lock };
    auto& map = mapForType(type);
    auto it = map.find(host);
    if (it == map.end())
        return std::nullopt;

    auto& response = it->value;
    if (response.expirationTime <= MonotonicTime::now()) {
        map.remove(it);
        return std::nullopt;
    }

    return response.addressList;
}

void DNSCache::update(const CString& host, Vector<GRefPtr<GInetAddress>>&& addressList, Type type)
{
    Locker locker { m_lock };
    auto& map = mapForType(type);
    CachedResponse response = { WTFMove(addressList), MonotonicTime::now() + expireInterval };
    auto addResult = map.set(host, WTFMove(response));
    if (addResult.isNewEntry)
        pruneResponsesInMap(map);
    m_expiredTimer.startOneShot(expireInterval);
}

void DNSCache::removeExpiredResponsesInMap(DNSCacheMap& map)
{
    map.removeIf([now = MonotonicTime::now()](auto& entry) {
        return entry.value.expirationTime <= now;
    });
}

void DNSCache::pruneResponsesInMap(DNSCacheMap& map)
{
    if (map.size() <= maxCacheSize)
        return;

    // First try to remove expired responses.
    removeExpiredResponsesInMap(map);
    if (map.size() <= maxCacheSize)
        return;

    Vector<CString> keys = copyToVector(map.keys());
    std::sort(keys.begin(), keys.end(), [&map](const CString& a, const CString& b) {
        return map.get(a).expirationTime < map.get(b).expirationTime;
    });

    unsigned responsesToRemoveCount = keys.size() - maxCacheSize;
    for (unsigned i = 0; i < responsesToRemoveCount; ++i)
        map.remove(keys[i]);
}

void DNSCache::removeExpiredResponsesFired()
{
    Locker locker { m_lock };
    removeExpiredResponsesInMap(m_dnsMap);
    removeExpiredResponsesInMap(m_ipv4Map);
    removeExpiredResponsesInMap(m_ipv6Map);
}

void DNSCache::clear()
{
    Locker locker { m_lock };
    m_dnsMap.clear();
    m_ipv4Map.clear();
    m_ipv6Map.clear();
}

} // namespace WebKit
