/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

#include <wtf/IndexedRange.h>

namespace WTF {

template<typename IteratorA, typename IteratorB> class ZippedRangeIterator {
public:
    ZippedRangeIterator(IteratorA&& iteratorA, IteratorB&& iteratorB)
        : m_iteratorA(WTFMove(iteratorA))
        , m_iteratorB(WTFMove(iteratorB))
    {
    }

    ZippedRangeIterator& operator++()
    {
        ++m_iteratorA;
        ++m_iteratorB;
        return *this;
    }

    auto operator*()
    {
        return std::pair<decltype(*m_iteratorA), decltype(*m_iteratorB)> { *m_iteratorA, *m_iteratorB };
    }

    bool operator==(const ZippedRangeIterator& other) const
    {
        // To ensure that we compare equal to end() even when we iterate two
        // collections of different sizes, we need to compare both A and B.
        // (Otherwise comparing either A or B would be sufficient, since they
        // increment in lockstep.)
        return m_iteratorA == other.m_iteratorA || m_iteratorB == other.m_iteratorB;
    }

private:
    IteratorA m_iteratorA;
    IteratorB m_iteratorB;
};

template<typename Iterator>
class ZippedRange {
public:
    template<typename CollectionA, typename CollectionB>
    ZippedRange(CollectionA&& collectionA, CollectionB&& collectionB)
        : m_begin(boundsCheckedBegin(std::forward<CollectionA>(collectionA)), boundsCheckedBegin(std::forward<CollectionB>(collectionB)))
        , m_end(boundsCheckedEnd(std::forward<CollectionA>(collectionA)), boundsCheckedEnd(std::forward<CollectionB>(collectionB)))
    {
        // Prevent use after destruction of a returned temporary.
        static_assert(std::ranges::borrowed_range<CollectionA>);
        static_assert(std::ranges::borrowed_range<CollectionB>);
    }

    auto begin() { return m_begin; }
    auto end() { return m_end; }

private:
    Iterator m_begin;
    Iterator m_end;
};

// Usage: for (auto [ valueA, valueB ] : zippedRange(collectionA, collectionB)) { ... }
template<typename CollectionA, typename CollectionB> auto zippedRange(CollectionA&& collectionA, CollectionB&& collectionB)
{
    using Iterator = ZippedRangeIterator<decltype(boundsCheckedBegin(std::forward<CollectionA>(collectionA))), decltype(boundsCheckedBegin(std::forward<CollectionB>(collectionB)))>;
    return ZippedRange<Iterator>(std::forward<CollectionA>(collectionA), std::forward<CollectionB>(collectionB));
}

} // namespace WTF

using WTF::zippedRange;
