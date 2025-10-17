/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include <array>
#include <iterator>
#include <optional>
#include <unicode/umachine.h>
#include <utility>

namespace PAL {

const std::array<std::pair<uint16_t, UChar>, 7724>& jis0208();
const std::array<std::pair<uint16_t, UChar>, 6067>& jis0212();
const std::array<std::pair<uint16_t, char32_t>, 18590>& big5();
const std::array<std::pair<uint16_t, UChar>, 17048>& eucKR();
const std::array<UChar, 23940>& gb18030();

void checkEncodingTableInvariants();

// Functions for using sorted arrays of pairs as a map.
// FIXME: Consider moving these functions to StdLibExtras.h for uses other than encoding tables.
template<typename CollectionType> void sortByFirst(CollectionType&);
template<typename CollectionType> void stableSortByFirst(CollectionType&);
template<typename CollectionType> bool isSortedByFirst(const CollectionType&);
template<typename CollectionType> bool sortedFirstsAreUnique(const CollectionType&);
template<typename CollectionType, typename KeyType> static auto findFirstInSortedPairs(const CollectionType& sortedPairsCollection, const KeyType&) -> std::optional<decltype(std::begin(sortedPairsCollection)->second)>;
template<typename CollectionType, typename KeyType> static auto findInSortedPairs(const CollectionType& sortedPairsCollection, const KeyType&) -> std::span<std::remove_reference_t<decltype(*std::begin(sortedPairsCollection))>>;

#if !ASSERT_ENABLED
inline void checkEncodingTableInvariants() { }
#endif

struct CompareFirst {
    template<typename TypeA, typename TypeB> bool operator()(const TypeA& a, const TypeB& b)
    {
        return a.first < b.first;
    }
};

struct EqualFirst {
    template<typename TypeA, typename TypeB> bool operator()(const TypeA& a, const TypeB& b)
    {
        return a.first == b.first;
    }
};

struct CompareSecond {
    template<typename TypeA, typename TypeB> bool operator()(const TypeA& a, const TypeB& b)
    {
        return a.second < b.second;
    }
};

template<typename T> struct FirstAdapter {
    const T& first;
};
template<typename T> FirstAdapter<T> makeFirstAdapter(const T& value)
{
    return { value };
}

template<typename T> struct SecondAdapter {
    const T& second;
};
template<typename T> SecondAdapter<T> makeSecondAdapter(const T& value)
{
    return { value };
}

template<typename CollectionType> void sortByFirst(CollectionType& collection)
{
    std::sort(std::begin(collection), std::end(collection), CompareFirst { });
}

template<typename CollectionType> void stableSortByFirst(CollectionType& collection)
{
    std::stable_sort(std::begin(collection), std::end(collection), CompareFirst { });
}

template<typename CollectionType> bool isSortedByFirst(const CollectionType& collection)
{
    return std::is_sorted(std::begin(collection), std::end(collection), CompareFirst { });
}

template<typename CollectionType> bool sortedFirstsAreUnique(const CollectionType& collection)
{
    return std::adjacent_find(std::begin(collection), std::end(collection), EqualFirst { }) == std::end(collection);
}

template<typename CollectionType, typename KeyType> static auto findFirstInSortedPairs(const CollectionType& collection, const KeyType& key) -> std::optional<decltype(std::begin(collection)->second)>
{
    if constexpr (std::is_integral_v<KeyType>) {
        if (key != decltype(std::begin(collection)->first)(key))
            return std::nullopt;
    }
    auto iterator = std::lower_bound(std::begin(collection), std::end(collection), makeFirstAdapter(key), CompareFirst { });
    if (iterator == std::end(collection) || key < iterator->first)
        return std::nullopt;
    return iterator->second;
}

template<typename CollectionType, typename KeyType> static auto findInSortedPairs(const CollectionType& collection, const KeyType& key) -> std::span<std::remove_reference_t<decltype(*std::begin(collection))>> {
    if constexpr (std::is_integral_v<KeyType>) {
        if (key != decltype(std::begin(collection)->first)(key))
            return { };
    }
    return std::ranges::equal_range(collection, makeFirstAdapter(key), CompareFirst { });
}

}
