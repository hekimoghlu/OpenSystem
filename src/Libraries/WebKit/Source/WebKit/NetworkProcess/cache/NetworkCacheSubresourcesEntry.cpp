/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)
#include "NetworkCacheSubresourcesEntry.h"

#include "Logging.h"
#include "NetworkCacheCoders.h"
#include <WebCore/RegistrableDomain.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WebKit {
namespace NetworkCache {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SubresourceInfo);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SubresourceLoad);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SubresourcesEntry);

bool SubresourceInfo::isFirstParty() const
{
    WebCore::RegistrableDomain firstPartyDomain { m_firstPartyForCookies };
    return firstPartyDomain.matches(URL(URL(), key().identifier()));
}

Storage::Record SubresourcesEntry::encodeAsStorageRecord() const
{
    WTF::Persistence::Encoder encoder;
    encoder << m_subresources;

    encoder.encodeChecksum();

    return { m_key, m_timeStamp, encoder.span(), { }, { } };
}

std::unique_ptr<SubresourcesEntry> SubresourcesEntry::decodeStorageRecord(const Storage::Record& storageEntry)
{
    auto entry = makeUnique<SubresourcesEntry>(storageEntry);

    WTF::Persistence::Decoder decoder(storageEntry.header.span());
    std::optional<Vector<SubresourceInfo>> subresources;
    decoder >> subresources;
    if (!subresources)
        return nullptr;
    entry->m_subresources = WTFMove(*subresources);

    if (!decoder.verifyChecksum()) {
        LOG(NetworkCache, "(NetworkProcess) checksum verification failure\n");
        return nullptr;
    }

    return entry;
}

SubresourcesEntry::SubresourcesEntry(const Storage::Record& storageEntry)
    : m_key(storageEntry.key)
    , m_timeStamp(storageEntry.timeStamp)
{
    ASSERT(m_key.type() == "SubResources"_s);
}

SubresourceInfo::SubresourceInfo(const Key& key, const WebCore::ResourceRequest& request, const SubresourceInfo* previousInfo)
    : m_key(key)
    , m_lastSeen(WallTime::now())
    , m_firstSeen(previousInfo ? previousInfo->firstSeen() : m_lastSeen)
    , m_isTransient(!previousInfo)
    , m_isSameSite(request.isSameSite())
    , m_isAppInitiated(request.isAppInitiated())
    , m_firstPartyForCookies(request.firstPartyForCookies())
    , m_requestHeaders(request.httpHeaderFields())
    , m_priority(request.priority())
{
}

static Vector<SubresourceInfo> makeSubresourceInfoVector(const Vector<std::unique_ptr<SubresourceLoad>>& subresourceLoads, Vector<SubresourceInfo>* previousSubresources)
{
    Vector<SubresourceInfo> result;
    result.reserveInitialCapacity(subresourceLoads.size());
    
    HashMap<Key, unsigned> previousMap;
    if (previousSubresources) {
        for (unsigned i = 0; i < previousSubresources->size(); ++i)
            previousMap.add(previousSubresources->at(i).key(), i);
    }

    HashSet<Key> deduplicationSet;
    for (auto& load : subresourceLoads) {
        if (!deduplicationSet.add(load->key).isNewEntry)
            continue;
        
        SubresourceInfo* previousInfo = nullptr;
        if (previousSubresources) {
            auto it = previousMap.find(load->key);
            if (it != previousMap.end())
                previousInfo = &(*previousSubresources)[it->value];
        }
        
        result.append({ load->key, load->request, previousInfo });
        
        // FIXME: We should really consider all resources seen for the first time transient.
        if (!previousSubresources)
            result.last().setNonTransient();
    }

    return result;
}

SubresourcesEntry::SubresourcesEntry(Key&& key, const Vector<std::unique_ptr<SubresourceLoad>>& subresourceLoads)
    : m_key(WTFMove(key))
    , m_timeStamp(WallTime::now())
    , m_subresources(makeSubresourceInfoVector(subresourceLoads, nullptr))
{
    ASSERT(m_key.type() == "SubResources"_s);
}
    
void SubresourcesEntry::updateSubresourceLoads(const Vector<std::unique_ptr<SubresourceLoad>>& subresourceLoads)
{
    m_subresources = makeSubresourceInfoVector(subresourceLoads, &m_subresources);
}

} // namespace WebKit
} // namespace NetworkCache

#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)
