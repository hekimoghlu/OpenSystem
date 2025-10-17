/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include <variant>
#include <wtf/HashMap.h>
#include <wtf/ScopedLambda.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

// This is a map optimized for holding 0 or 1 items with no hashing or allocations in those cases.
template<typename Key, typename Value>
class SmallMap {
public:
    using Pair = KeyValuePair<Key, Value>;
    using Map = UncheckedKeyHashMap<Key, Value>;
    using Storage = std::variant<std::monostate, Pair, Map>;

    static_assert(sizeof(Pair) <= 4 * sizeof(uint64_t), "Don't use SmallMap with large types. It probably wastes memory.");

    Value& ensure(const Key& key, const auto& functor)
    {
        ASSERT(Map::isValidKey(key));
        if (std::holds_alternative<std::monostate>(m_storage)) {
            m_storage = Pair { key, functor() };
            return std::get<Pair>(m_storage).value;
        }
        if (auto* pair = std::get_if<Pair>(&m_storage)) {
            if (pair->key == key)
                return pair->value;
            Map map;
            map.add(WTFMove(pair->key), WTFMove(pair->value));
            m_storage = WTFMove(map);
            return std::get<Map>(m_storage).add(key, functor()).iterator->value;
        }
        return std::get<Map>(m_storage).ensure(key, functor).iterator->value;
    }

    void remove(const Key& key)
    {
        ASSERT(Map::isValidKey(key));
        if (auto* pair = std::get_if<Pair>(&m_storage)) {
            if (pair->key == key)
                m_storage = std::monostate { };
        } else if (auto* map = std::get_if<Map>(&m_storage))
            map->remove(key);
    }

    const Value* get(const Key& key) const
    {
        ASSERT(Map::isValidKey(key));
        if (auto* pair = std::get_if<Pair>(&m_storage)) {
            if (pair->key == key)
                return std::addressof(pair->value);
        } else if (auto* map = std::get_if<Map>(&m_storage)) {
            if (auto it = map->find(key); it != map->end())
                return std::addressof(it->value);
        }
        return nullptr;
    }

    void forEach(const auto& callback) const
    {
        switchOn(m_storage, [&] (const std::monostate&) {
        }, [&] (const Pair& pair) {
            callback(pair.key, pair.value);
        }, [&] (const Map& map) {
            for (auto& [key, value] : map)
                callback(key, value);
        });
    }

    size_t size() const
    {
        return switchOn(m_storage, [&] (const std::monostate&) {
            return 0u;
        }, [&] (const Pair&) {
            return 1u;
        }, [&] (const Map& map) {
            return map.size();
        });
    }

    const Storage& rawStorage() const { return m_storage; }

private:
    Storage m_storage;
};

} // namespace WTF

using WTF::SmallMap;
