/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#include <wtf/Assertions.h>
#include <wtf/BitVector.h>
#include <wtf/FastMalloc.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(LikelyDenseUnsignedIntegerSet);

// This is effectively a std::variant<HashSet, Pair<BitVector, IndexType>>
// If it is in BitVector mode, it keeps track of the minimum value in the set, and has the bitVector shifted by the same amount.
// So for example {64000, 64002, 64003} would be represented as the bitVector 1101 with a m_min of 64000.
// It shifts between the two modes whenever that would at least halve its memory usage. So it will never use more than twice the optimal amount of memory, and yet should not ping-pong between the two modes too often.
// As an optimization, instead of keeping track of the minimum value, it keeps track of the minimum value rounded down to the next multiple of 64.
// This reduces repeated re-indexings of the bitvector when repeatedly adding a value just below the current minimum.
template<typename IndexType>
class LikelyDenseUnsignedIntegerSet {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(LikelyDenseUnsignedIntegerSet);
    static_assert(std::is_unsigned<IndexType>::value);
    using Set = UncheckedKeyHashSet<IndexType, WTF::IntHash<IndexType>, WTF::UnsignedWithZeroKeyHashTraits<IndexType> >;
public:
    LikelyDenseUnsignedIntegerSet()
        : m_size(0)
        , m_min(0)
        , m_max(0)
    {
        new (NotNull, &m_inline.bitVector) BitVector;
    }

    ~LikelyDenseUnsignedIntegerSet()
    {
        if (isBitVector())
            m_inline.bitVector.~BitVector();
        else
            m_inline.hashSet.~Set();
    }

    LikelyDenseUnsignedIntegerSet(LikelyDenseUnsignedIntegerSet&& other)
        : m_size(other.m_size)
        , m_min(other.m_min)
        , m_max(other.m_max)
    {
        if (isBitVector())
            new (NotNull, &m_inline.bitVector) BitVector(WTFMove(other.m_inline.bitVector));
        else
            new (NotNull, &m_inline.hashSet) Set(WTFMove(other.m_inline.hashSet));
    }

    void clear()
    {
        if (isBitVector())
            m_inline.bitVector.~BitVector();
        else
            m_inline.hashSet.~HashSet();
        new (NotNull, &m_inline.bitVector) BitVector;
        m_size = 0;
        m_min = 0;
        m_max = 0;
    }

    bool contains(IndexType value) const
    {
        ASSERT(isValidValue(value));

        if (isBitVector()) {
            if (m_min > value)
                return false;
            return m_inline.bitVector.get(value - m_min);
        }

        return m_inline.hashSet.contains(value);
    }

    // Providing an iterator in this would be possible, but none of our clients need it.
    struct AddResult {
        bool isNewEntry;
    };

    AddResult add(IndexType value)
    {
        ASSERT(isValidValue(value));

        if (!m_size) {
            ASSERT(isBitVector());
            m_min = value & ~63;
            m_max = value;
            m_size = 1;
            // not quickSet, as value - m_min might be 63, and the inline bit vector cannot store that value.
            // So there might be some overflow here, forcing an allocation of an outline bit vector.
            m_inline.bitVector.set(value - m_min);
            return { true };
        }

        auto computeNewMin = [&]() {
            IndexType roundedDownValue = value & ~63;
            return std::min(m_min, roundedDownValue);
        };

        if (!isBitVector()) {
            bool isNewEntry = m_inline.hashSet.add(value).isNewEntry;
            if (!isNewEntry)
                return { false };
            m_min = computeNewMin();
            m_max = std::max(m_max, value);
            unsigned hashSetSize = m_inline.hashSet.capacity() * sizeof(IndexType);
            unsigned wouldBeBitVectorSize = (m_max - m_min) / 8;
            if (wouldBeBitVectorSize * 2 < hashSetSize)
                transitionToBitVector();
            return { true };
        }

        if (value >= m_min && value <= m_max) {
            bool isNewEntry = !m_inline.bitVector.quickSet(value - m_min);
            m_size += isNewEntry;
            return { isNewEntry };
        }

        // We are in BitVector mode, and value is not in the bounds: we will definitely insert it as a new entry.
        ++m_size;

        IndexType newMin = computeNewMin();
        IndexType newMax = std::max(m_max, value);
        unsigned bitVectorSize = (newMax - newMin) / 8;
        unsigned wouldBeHashSetSize = estimateHashSetSize(m_size);
        if (wouldBeHashSetSize * 2 < bitVectorSize) {
            transitionToHashSet();
            auto result = m_inline.hashSet.add(value);
            ASSERT_UNUSED(result, result.isNewEntry);
            m_min = newMin;
            m_max = newMax;
            return { true };
        }

        if (value < m_min) {
            ASSERT(newMin < m_min);
            m_inline.bitVector.shiftRightByMultipleOf64(m_min - newMin);
            m_min = newMin;
        }

        bool isNewEntry = !m_inline.bitVector.set(value - m_min);
        ASSERT_UNUSED(isNewEntry, isNewEntry);
        m_max = newMax;
        return { true };
    }

    unsigned size() const
    {
        if (isBitVector())
            return m_size;
        return m_inline.hashSet.size();
    }

    class iterator {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        iterator(BitVector::iterator it, IndexType shift)
            : m_underlying({ it })
            , m_shift(shift)
        {
        }
        iterator(typename Set::iterator it, IndexType shift)
            : m_underlying({ it })
            , m_shift(shift)
        {
        }

        iterator& operator++()
        {
            WTF::switchOn(m_underlying,
                [](BitVector::iterator& it) { ++it; },
                [](typename Set::iterator& it) { ++it; });
            return *this;
        }

        IndexType operator*() const
        {
            return WTF::switchOn(m_underlying,
                [&](const BitVector::iterator& it) -> IndexType { return *it + m_shift; },
                [](const typename Set::iterator& it) -> IndexType { return *it; });
        }

        friend bool operator==(const iterator&, const iterator&) = default;

    private:
        std::variant<BitVector::iterator, typename Set::iterator> m_underlying;
        IndexType m_shift;
    };

    iterator begin() const
    {
        if (isBitVector())
            return { m_inline.bitVector.begin(), m_min };
        return { m_inline.hashSet.begin(), m_min };
    }
    iterator end() const
    {
        if (isBitVector())
            return { m_inline.bitVector.end(), m_min };
        return { m_inline.hashSet.end(), m_min };
    }

    unsigned memoryUse() const
    {
        unsigned result = sizeof(LikelyDenseUnsignedIntegerSet);
        if (isBitVector())
            result += m_inline.bitVector.outOfLineMemoryUse();
        else
            result += m_inline.hashSet.capacity() * sizeof(IndexType);
        return result;
    }

    void validate() const
    {
        IndexType min = std::numeric_limits<IndexType>::max(), max = 0;
        if (isBitVector()) {
            unsigned numElements = 0;
            for (IndexType shiftedIndex : m_inline.bitVector) {
                ++numElements;
                IndexType correctedIndex = m_min + shiftedIndex;
                min = std::min(min, correctedIndex);
                max = std::max(max, correctedIndex);
            }
            RELEASE_ASSERT(m_size == numElements);
        } else {
            for (IndexType index : m_inline.hashSet) {
                min = std::min(min, index);
                max = std::max(max, index);
            }
        }
        if (m_size) {
            RELEASE_ASSERT(m_min <= min);
            RELEASE_ASSERT(m_min + 64 > min);
            RELEASE_ASSERT(!(m_min & 63));
            RELEASE_ASSERT(m_max == max);
        }
    }

private:
    bool isBitVector() const { return m_size != std::numeric_limits<unsigned>::max(); }

    bool isValidValue(IndexType value) const
    {
        return value != UnsignedWithZeroKeyHashTraits<IndexType>::emptyValue()
            && !UnsignedWithZeroKeyHashTraits<IndexType>::isDeletedValue(value);
    }

    unsigned estimateHashSetSize(unsigned n)
    {
        unsigned hashSetEstimatedOccupancyOverhead = 3; // max occupancy for a large HashSet is 50%
        unsigned wouldBeHashSetCapacity = std::max(8u, n) * hashSetEstimatedOccupancyOverhead;
        return wouldBeHashSetCapacity * sizeof(IndexType);
    }

    void transitionToHashSet()
    {
        ASSERT(isBitVector());

        Set newSet;
        newSet.reserveInitialCapacity(m_size + 1);
        for (IndexType oldIndex : m_inline.bitVector)
            newSet.add(oldIndex + m_min);
        m_inline.bitVector.~BitVector();
        new (NotNull, &m_inline.hashSet) Set(WTFMove(newSet));
        m_size = std::numeric_limits<unsigned>::max();

        ASSERT(!isBitVector());
    }

    void transitionToBitVector()
    {
        ASSERT(!isBitVector());

        BitVector newBitVector;
        newBitVector.ensureSize(m_max - m_min + 1);
        m_size = 0;
        for (IndexType oldValue : m_inline.hashSet) {
            newBitVector.quickSet(oldValue - m_min);
            ++m_size;
        }
        m_inline.hashSet.~Set();
        new (NotNull, &m_inline.bitVector) BitVector(WTFMove(newBitVector));

        ASSERT(isBitVector());
    }

    union U {
        BitVector bitVector;
        Set hashSet;

        U() { }
        ~U() { }
    } m_inline;
    unsigned m_size; // Only updated in BitVector mode, std::numeric_limits<unsigned>::max() to indicate HashSet mode
    IndexType m_min;
    IndexType m_max;

};

} // namespace WTF

using WTF::LikelyDenseUnsignedIntegerSet;
