/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include <wtf/Forward.h>
#include <wtf/HashTraits.h>

namespace WTF {

template<typename Key, typename HashArg = DefaultHash<Key>>
class GenericHashKey final {
    WTF_MAKE_FAST_ALLOCATED;

    struct EmptyKey { };
    struct DeletedKey { };

public:
    constexpr GenericHashKey(Key&& key)
        : m_value(std::in_place_type_t<Key>(), WTFMove(key))
    {
    }

    template<typename K>
    constexpr GenericHashKey(K&& key)
        : m_value(std::in_place_type_t<Key>(), std::forward<K>(key))
    {
    }

    constexpr GenericHashKey(HashTableEmptyValueType)
        : m_value(EmptyKey { })
    {
    }

    constexpr GenericHashKey(HashTableDeletedValueType)
        : m_value(DeletedKey { })
    {
    }

    constexpr const Key& key() const { return std::get<Key>(m_value); }
    constexpr unsigned hash() const
    {
        ASSERT_UNDER_CONSTEXPR_CONTEXT(!isHashTableDeletedValue() && !isHashTableEmptyValue());
        return HashArg::hash(key());
    }

    constexpr bool isHashTableDeletedValue() const { return std::holds_alternative<DeletedKey>(m_value); }
    constexpr bool isHashTableEmptyValue() const { return std::holds_alternative<EmptyKey>(m_value); }

    constexpr bool operator==(const GenericHashKey& other) const
    {
        if (m_value.index() != other.m_value.index())
            return false;
        if (!std::holds_alternative<Key>(m_value))
            return true;
        return HashArg::equal(key(), other.key());
    }

private:
    std::variant<Key, EmptyKey, DeletedKey> m_value;
};

template<typename K, typename H> struct HashTraits<GenericHashKey<K, H>> : GenericHashTraits<GenericHashKey<K, H>> {
    static GenericHashKey<K, H> emptyValue() { return GenericHashKey<K, H> { HashTableEmptyValue }; }
    static bool isEmptyValue(const GenericHashKey<K, H>& value) { return value.isHashTableEmptyValue(); }
    static void constructDeletedValue(GenericHashKey<K, H>& slot) { slot = GenericHashKey<K, H> { HashTableDeletedValue }; }
    static bool isDeletedValue(const GenericHashKey<K, H>& value) { return value.isHashTableDeletedValue(); }
};

template<typename K, typename H> struct DefaultHash<GenericHashKey<K, H>> {
    static unsigned hash(const GenericHashKey<K, H>& key) { return key.hash(); }
    static bool equal(const GenericHashKey<K, H>& a, const GenericHashKey<K, H>& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

}

using WTF::GenericHashKey;
