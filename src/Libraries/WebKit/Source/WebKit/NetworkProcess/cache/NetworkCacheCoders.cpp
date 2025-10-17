/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "NetworkCacheCoders.h"

#include "NetworkCacheKey.h"
#include "NetworkCacheSubresourcesEntry.h"
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/persistence/PersistentDecoder.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WTF::Persistence {

void Coder<WebKit::NetworkCache::Key>::encodeForPersistence(WTF::Persistence::Encoder& encoder, const WebKit::NetworkCache::Key& instance)
{
    encoder << instance.partition();
    encoder << instance.type();
    encoder << instance.identifier();
    encoder << instance.range();
    encoder << instance.hash();
    encoder << instance.partitionHash();
}

std::optional<WebKit::NetworkCache::Key> Coder<WebKit::NetworkCache::Key>::decodeForPersistence(WTF::Persistence::Decoder& decoder)
{
    WebKit::NetworkCache::Key key;

    std::optional<String> partition;
    decoder >> partition;
    if (!partition)
        return std::nullopt;
    key.m_partition = WTFMove(*partition);

    std::optional<String> type;
    decoder >> type;
    if (!type)
        return std::nullopt;
    key.m_type = WTFMove(*type);

    std::optional<String> identifier;
    decoder >> identifier;
    if (!identifier)
        return std::nullopt;
    key.m_identifier = WTFMove(*identifier);

    std::optional<String> range;
    decoder >> range;
    if (!range)
        return std::nullopt;
    key.m_range = WTFMove(*range);

    std::optional<WebKit::NetworkCache::Key::HashType> hash;
    decoder >> hash;
    if (!hash)
        return std::nullopt;
    key.m_hash = WTFMove(*hash);

    std::optional<WebKit::NetworkCache::Key::HashType> partitionHash;
    decoder >> partitionHash;
    if (!partitionHash)
        return std::nullopt;
    key.m_partitionHash = WTFMove(*partitionHash);

    return { WTFMove(key) };
}

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)
void Coder<WebKit::NetworkCache::SubresourceInfo>::encodeForPersistence(WTF::Persistence::Encoder& encoder, const WebKit::NetworkCache::SubresourceInfo& instance)
{
    encoder << instance.key();
    encoder << instance.lastSeen();
    encoder << instance.firstSeen();
    encoder << instance.isTransient();

    // Do not bother serializing other data members of transient resources as they are empty.
    if (instance.isTransient())
        return;

    encoder << instance.isSameSite();
    encoder << instance.isAppInitiated();
    encoder << instance.firstPartyForCookies();
    encoder << instance.requestHeaders();
    encoder << instance.priority();
}

std::optional<WebKit::NetworkCache::SubresourceInfo> Coder<WebKit::NetworkCache::SubresourceInfo>::decodeForPersistence(WTF::Persistence::Decoder& decoder)
{
    std::optional<WebKit::NetworkCache::Key> key;
    decoder >> key;
    if (!key)
        return std::nullopt;

    std::optional<WallTime> lastSeen;
    decoder >> lastSeen;
    if (!lastSeen)
        return std::nullopt;

    std::optional<WallTime> firstSeen;
    decoder >> firstSeen;
    if (!firstSeen)
        return std::nullopt;

    std::optional<bool> isTransient;
    decoder >> isTransient;
    if (!isTransient)
        return std::nullopt;

    if (*isTransient)
        return WebKit::NetworkCache::SubresourceInfo(WTFMove(*key), *lastSeen, *firstSeen);

    std::optional<bool> isSameSite;
    decoder >> isSameSite;
    if (!isSameSite)
        return std::nullopt;

    std::optional<bool> isAppInitiated;
    decoder >> isAppInitiated;
    if (!isAppInitiated)
        return std::nullopt;

    std::optional<URL> firstPartyForCookies;
    decoder >> firstPartyForCookies;
    if (!firstPartyForCookies)
        return std::nullopt;

    std::optional<WebCore::HTTPHeaderMap> requestHeaders;
    decoder >> requestHeaders;
    if (!requestHeaders)
        return std::nullopt;

    std::optional<WebCore::ResourceLoadPriority> priority;
    decoder >> priority;
    if (!priority)
        return std::nullopt;

    return WebKit::NetworkCache::SubresourceInfo(WTFMove(*key), *lastSeen, *firstSeen, *isSameSite, *isAppInitiated, WTFMove(*firstPartyForCookies), WTFMove(*requestHeaders), *priority);
}
#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)

}
