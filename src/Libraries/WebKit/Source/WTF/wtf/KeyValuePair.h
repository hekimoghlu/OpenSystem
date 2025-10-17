/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#include <type_traits>

namespace WTF {

template<typename KeyTypeArg, typename ValueTypeArg>
struct KeyValuePair {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    using KeyType = KeyTypeArg;
    using ValueType = ValueTypeArg;

    KeyValuePair()
    {
    }

    KeyValuePair(KeyTypeArg&& key, ValueTypeArg&& value)
        : key(WTFMove(key))
        , value(WTFMove(value))
    {
    }

    template<typename K, typename V>
    KeyValuePair(K&& key, V&& value)
        : key(std::forward<K>(key))
        , value(std::forward<V>(value))
    {
    }

    template <typename K, typename V>
    KeyValuePair(KeyValuePair<K, V>&& other)
        : key(std::forward<K>(other.key))
        , value(std::forward<V>(other.value))
    {
    }

    KeyType key;
    ValueType value { };
};

template<typename K, typename V>
inline KeyValuePair<typename std::decay<K>::type, typename std::decay<V>::type> makeKeyValuePair(K&& key, V&& value)
{
    return KeyValuePair<typename std::decay<K>::type, typename std::decay<V>::type> { std::forward<K>(key), std::forward<V>(value) };
}

template<typename KeyType, typename ValueType> constexpr bool operator==(const KeyValuePair<KeyType, ValueType>& a, const KeyValuePair<KeyType, ValueType>& b)
{
    return a.key == b.key && a.value == b.value;
}

}

using WTF::KeyValuePair;
