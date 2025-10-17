/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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

#include <algorithm>
#include <cstddef>
#include <span>
#include <wtf/NeverDestroyed.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

template<typename KeyType, typename ValueType>
struct TinyLRUCachePolicy {
    static bool isKeyNull(const KeyType&) { return false; }
    static ValueType createValueForNullKey() { return { }; }
    static ValueType createValueForKey(const KeyType&) { return { }; }
    static KeyType createKeyForStorage(const KeyType& key) { return key; }
};

template<typename KeyType, typename ValueType, size_t capacity = 4, typename Policy = TinyLRUCachePolicy<KeyType, ValueType>>
class TinyLRUCache {
    WTF_MAKE_FAST_ALLOCATED;
public:
    const ValueType& get(const KeyType& key)
    {
        if (Policy::isKeyNull(key)) {
            static NeverDestroyed<ValueType> valueForNull = Policy::createValueForNullKey();
            return valueForNull;
        }

        auto cacheBuffer = this->cacheBuffer();
        for (size_t i = m_size; i-- > 0;) {
            if (cacheBuffer[i].first == key) {
                if (i < m_size - 1) {
                    // Move entry to the end of the cache if necessary.
                    auto entry = WTFMove(cacheBuffer[i]);
                    do {
                        cacheBuffer[i] = WTFMove(cacheBuffer[i + 1]);
                    } while (++i < m_size - 1);
                    cacheBuffer[m_size - 1] = WTFMove(entry);
                }
                return cacheBuffer[m_size - 1].second;
            }
        }

        // cacheBuffer[0] is the LRU entry, so remove it.
        if (m_size == capacity) {
            for (size_t i = 0; i < m_size - 1; ++i)
                cacheBuffer[i] = WTFMove(cacheBuffer[i + 1]);
        } else
            ++m_size;

        cacheBuffer[m_size - 1] = std::pair { Policy::createKeyForStorage(key), Policy::createValueForKey(key) };
        return cacheBuffer[m_size - 1].second;
    }

private:
    using Entry = std::pair<KeyType, ValueType>;
    std::span<Entry, capacity> cacheBuffer() { return m_cacheBuffer; }

    alignas(Entry) std::array<Entry, capacity> m_cacheBuffer;
    size_t m_size { 0 };
};

} // namespace WTF

using WTF::TinyLRUCache;
using WTF::TinyLRUCachePolicy;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
