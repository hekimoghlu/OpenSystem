/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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

namespace WTF {

template<typename Iterator> class BoundsCheckedIterator {
public:
    // We require the caller to ask for 'begin' or 'end', rather than passing
    // us arbitrary 'it' and 'end' iterators, because that way we can prove by
    // construction that we have the correct 'end'.
    template<typename Collection>
    static BoundsCheckedIterator begin(Collection&& collection)
    {
        return BoundsCheckedIterator(std::forward<Collection>(collection), std::begin(collection));
    }

    template<typename Collection>
    static BoundsCheckedIterator end(Collection&& collection)
    {
        return BoundsCheckedIterator(std::forward<Collection>(collection), std::end(collection));
    }

    BoundsCheckedIterator& operator++()
    {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        RELEASE_ASSERT(m_iterator != m_end);
        ++m_iterator;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

        return *this;
    }

    auto&& operator*() const
    {
        RELEASE_ASSERT(m_iterator != m_end);
        return *m_iterator;
    }

    bool operator==(const BoundsCheckedIterator& other) const
    {
        return m_iterator == other.m_iterator;
    }

private:
    template<typename Collection>
    BoundsCheckedIterator(Collection&& collection, Iterator&& iterator)
        : m_iterator(WTFMove(iterator))
        , m_end(std::end(collection))
    {
    }

    Iterator m_iterator;
    Iterator m_end;
};

template<typename Collection> auto boundsCheckedBegin(Collection&& collection)
{
    return BoundsCheckedIterator<decltype(std::begin(collection))>::begin(std::forward<Collection>(collection));
}

template<typename Collection> auto boundsCheckedEnd(Collection&& collection)
{
    return BoundsCheckedIterator<decltype(std::end(collection))>::end(std::forward<Collection>(collection));
}

template<typename Iterator> class IndexedRangeIterator {
public:
    IndexedRangeIterator(Iterator&& iterator)
        : m_iterator(WTFMove(iterator))
    {
    }

    IndexedRangeIterator& operator++()
    {
        ++m_index;
        ++m_iterator;
        return *this;
    }

    auto operator*()
    {
        return std::pair<size_t, decltype(*m_iterator)> { m_index, *m_iterator };
    }

    bool operator==(const IndexedRangeIterator& other) const
    {
        return m_iterator == other.m_iterator;
    }

private:
    size_t m_index { 0 };
    Iterator m_iterator;
};

template<typename Iterator>
class IndexedRange {
public:
    template<typename Collection>
    IndexedRange(Collection&& collection)
        : m_begin(boundsCheckedBegin(std::forward<Collection>(collection)))
        , m_end(boundsCheckedEnd(std::forward<Collection>(collection)))
    {
        // Prevent use after destruction of a returned temporary.
        static_assert(std::ranges::borrowed_range<Collection>);
    }

    auto begin() { return m_begin; }
    auto end() { return m_end; }

private:
    Iterator m_begin;
    Iterator m_end;
};

// Usage: for (auto [ index, value ] : indexedRange(collection)) { ... }
template<typename Collection> auto indexedRange(Collection&& collection)
{
    using Iterator = IndexedRangeIterator<decltype(boundsCheckedBegin(std::forward<Collection>(collection)))>;
    return IndexedRange<Iterator>(std::forward<Collection>(collection));
}

} // namespace WTF

using WTF::indexedRange;
