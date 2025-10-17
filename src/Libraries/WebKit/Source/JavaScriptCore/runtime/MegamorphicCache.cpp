/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#include "MegamorphicCache.h"

#include <wtf/TZoneMallocInlines.h>

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(MegamorphicCache);
WTF_MAKE_TZONE_ALLOCATED_IMPL(MegamorphicCache);

void MegamorphicCache::age(CollectionScope collectionScope)
{
    ++m_epoch;
    if (collectionScope == CollectionScope::Full || m_epoch == invalidEpoch) {
        for (auto& entry : m_loadCachePrimaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        for (auto& entry : m_loadCacheSecondaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        for (auto& entry : m_storeCachePrimaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        for (auto& entry : m_storeCacheSecondaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        for (auto& entry : m_hasCachePrimaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        for (auto& entry : m_hasCacheSecondaryEntries) {
            entry.m_uid = nullptr;
            entry.m_epoch = invalidEpoch;
        }
        if (m_epoch == invalidEpoch)
            m_epoch = 1;
    }
}

void MegamorphicCache::clearEntries()
{
    for (auto& entry : m_loadCachePrimaryEntries)
        entry.m_epoch = invalidEpoch;
    for (auto& entry : m_loadCacheSecondaryEntries)
        entry.m_epoch = invalidEpoch;
    for (auto& entry : m_storeCachePrimaryEntries)
        entry.m_epoch = invalidEpoch;
    for (auto& entry : m_storeCacheSecondaryEntries)
        entry.m_epoch = invalidEpoch;
    for (auto& entry : m_hasCachePrimaryEntries)
        entry.m_epoch = invalidEpoch;
    for (auto& entry : m_hasCacheSecondaryEntries)
        entry.m_epoch = invalidEpoch;
    m_epoch = 1;
}

} // namespace JSC
