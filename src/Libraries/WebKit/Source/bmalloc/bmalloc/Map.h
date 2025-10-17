/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

BALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "BInline.h"
#include "Sizes.h"
#include "Vector.h"
#include <optional>

namespace bmalloc {

class SmallPage;

enum class AllowDeleting { DeletingAllowed, DeletingNotAllowed };

template<typename Key, typename Value, typename Hash, AllowDeleting allowDeleting = AllowDeleting::DeletingNotAllowed> class Map {
    static_assert(std::is_trivially_destructible<Key>::value, "Map must have a trivial destructor.");
    static_assert(std::is_trivially_destructible<Value>::value, "Map must have a trivial destructor.");
public:
    struct Bucket {
        Key key;
        Value value;
    };

    Map();

    size_t size() { return m_keyCount; }
    size_t capacity() { return m_table.size(); }

    // key must be in the map.
    Value& get(const Key& key)
    {
        auto& bucket = find(key, [&](const Bucket& bucket) { return bucket.key == key; });
        return bucket.value;
    }

    std::optional<Value> getOptional(const Key& key)
    {
        if (!size())
            return std::nullopt;

        auto& bucket = find(key, [&](const Bucket& bucket) {
            return allowDeleting == AllowDeleting::DeletingAllowed ? bucket.key == key : !bucket.key || bucket.key == key;
        });

        if (bucket.key)
            return bucket.value;
        return std::nullopt;
    }

    void set(const Key& key, const Value& value)
    {
        if (shouldGrow())
            rehash();

        auto& bucket = find(key, [&](const Bucket& bucket) {
            return allowDeleting == AllowDeleting::DeletingAllowed ? bucket.key == key : !bucket.key || bucket.key == key;
        });
        if (!bucket.key) {
            bucket.key = key;
            ++m_keyCount;
        }
        bucket.value = value;
    }

    bool contains(const Key& key)
    {
        if (!size())
            return false;

        auto& bucket = find(key, [&](const Bucket& bucket) {
            return allowDeleting == AllowDeleting::DeletingAllowed ? bucket.key == key : !bucket.key || bucket.key == key;
        });

        return !!bucket.key;
    }

    // key must be in the map.
    Value remove(const Key& key)
    {
        RELEASE_BASSERT(allowDeleting == AllowDeleting::DeletingAllowed);

        if (shouldShrink())
            rehash();

        auto& bucket = find(key, [&](const Bucket& bucket) { return bucket.key == key; });
        Value value = bucket.value;
        bucket.key = Key();
        --m_keyCount;
        return value;
    }

private:
    static constexpr unsigned minCapacity = 16;
    static constexpr unsigned maxLoad = 2;
    static constexpr unsigned rehashLoad = 4;
    static constexpr unsigned minLoad = 8;

    bool shouldGrow() { return m_keyCount * maxLoad >= capacity(); }
    bool shouldShrink() { return m_keyCount * minLoad <= capacity() && capacity() > minCapacity; }

    void rehash();

    template<typename Predicate>
    Bucket& find(const Key& key, const Predicate& predicate)
    {
        unsigned keysChecked = 0;
        Bucket* firstEmptyBucket = nullptr;

        for (unsigned h = Hash::hash(key); ; ++h) {
            unsigned i = h & m_tableMask;

            Bucket& bucket = m_table[i];
            if (predicate(bucket))
                return bucket;
            if (allowDeleting == AllowDeleting::DeletingAllowed) {
                if (bucket.key)
                    ++keysChecked;
                else {
                    if (!firstEmptyBucket)
                        firstEmptyBucket = &bucket;

                    if (keysChecked >= m_keyCount) {
                        if (firstEmptyBucket)
                            return *firstEmptyBucket;
                        BASSERT(!bucket.key);
                        return bucket;
                    }
                }
            }
        }
    }

    unsigned m_keyCount;
    unsigned m_tableMask;
    Vector<Bucket> m_table;
};

template<typename Key, typename Value, typename Hash, enum AllowDeleting allowDeleting>
inline Map<Key, Value, Hash, allowDeleting>::Map()
    : m_keyCount(0)
    , m_tableMask(0)
{
}

template<typename Key, typename Value, typename Hash, enum AllowDeleting allowDeleting>
void Map<Key, Value, Hash, allowDeleting>::rehash()
{
    auto oldTable = std::move(m_table);

    size_t newCapacity = std::max(minCapacity, m_keyCount * rehashLoad);
    m_table.grow(newCapacity);

    m_keyCount = 0;
    m_tableMask = newCapacity - 1;

    for (auto& bucket : oldTable) {
        if (!bucket.key)
            continue;

        BASSERT(!shouldGrow());
        set(bucket.key, bucket.value);
    }
}

template<typename Key, typename Value, typename Hash, enum AllowDeleting allowDeleting> const unsigned Map<Key, Value, Hash, allowDeleting>::minCapacity;

} // namespace bmalloc

BALLOW_UNSAFE_BUFFER_USAGE_END
