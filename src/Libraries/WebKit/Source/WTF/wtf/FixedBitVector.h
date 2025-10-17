/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#include <wtf/Assertions.h>
#include <wtf/Atomics.h>
#include <wtf/BitVector.h>
#include <wtf/HashTraits.h>
#include <wtf/PrintStream.h>
#include <wtf/StdIntExtras.h>
#include <wtf/StdLibExtras.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// FIXME: This should be `: private BitVector`.
class FixedBitVector final {
    WTF_MAKE_FAST_ALLOCATED;
    using WordType = decltype(BitVector::m_bitsOrPointer);

public:
    FixedBitVector() = default;

    FixedBitVector(size_t size)
        : m_bitVector(size)
    {
    }

    FixedBitVector(BitVector&& other)
        : m_bitVector(WTFMove(other))
    {
    }

    bool concurrentTestAndSet(size_t bitIndex, Dependency = Dependency());
    bool concurrentTestAndClear(size_t bitIndex, Dependency = Dependency());

    bool testAndSet(size_t bitIndex);
    bool testAndClear(size_t bitIndex);
    bool test(size_t bitIndex);

    inline void merge(const FixedBitVector& other);
    inline void filter(const FixedBitVector& other);
    inline void exclude(const FixedBitVector& other);

    // Note that BitVector will be in inline mode with fixed size when
    // the BitVector is constructed with size less or equal to `maxInlineBits`.
    size_t size() const { return m_bitVector.size(); }
    size_t bitCount() const { return m_bitVector.bitCount(); }

    bool isEmpty() const { return m_bitVector.isEmpty(); }

    size_t findBit(size_t startIndex, bool value) const;

    friend bool operator==(const FixedBitVector&, const FixedBitVector&) = default;

    unsigned hash() const;

    void dump(PrintStream& out) const;

    BitVector::iterator begin() const { return m_bitVector.begin(); }
    BitVector::iterator end() const { return m_bitVector.end(); }

private:
    static constexpr unsigned wordSize = sizeof(WordType) * 8;
    static constexpr WordType one = 1;

    BitVector m_bitVector;
};

ALWAYS_INLINE bool FixedBitVector::concurrentTestAndSet(size_t bitIndex, Dependency dependency)
{
    if (UNLIKELY(bitIndex >= size()))
        return false;

    WordType mask = one << (bitIndex % wordSize);
    size_t wordIndex = bitIndex / wordSize;
    WordType* data = dependency.consume(m_bitVector.bits()) + wordIndex;
    return !std::bit_cast<Atomic<WordType>*>(data)->transactionRelaxed(
        [&](WordType& value) -> bool {
            if (value & mask)
                return false;

            value |= mask;
            return true;
        });
}

ALWAYS_INLINE bool FixedBitVector::concurrentTestAndClear(size_t bitIndex, Dependency dependency)
{
    if (UNLIKELY(bitIndex >= size()))
        return false;

    WordType mask = one << (bitIndex % wordSize);
    size_t wordIndex = bitIndex / wordSize;
    WordType* data = dependency.consume(m_bitVector.bits()) + wordIndex;
    return std::bit_cast<Atomic<WordType>*>(data)->transactionRelaxed(
        [&](WordType& value) -> bool {
            if (!(value & mask))
                return false;

            value &= ~mask;
            return true;
        });
}

ALWAYS_INLINE bool FixedBitVector::testAndSet(size_t bitIndex)
{
    if (UNLIKELY(bitIndex >= size()))
        return false;

    WordType mask = one << (bitIndex % wordSize);
    size_t wordIndex = bitIndex / wordSize;
    WordType* bits = m_bitVector.bits();
    bool previousValue = bits[wordIndex] & mask;
    bits[wordIndex] |= mask;
    return previousValue;
}

ALWAYS_INLINE bool FixedBitVector::testAndClear(size_t bitIndex)
{
    if (UNLIKELY(bitIndex >= size()))
        return false;

    WordType mask = one << (bitIndex % wordSize);
    size_t wordIndex = bitIndex / wordSize;
    WordType* bits = m_bitVector.bits();
    bool previousValue = bits[wordIndex] & mask;
    bits[wordIndex] &= ~mask;
    return previousValue;
}

ALWAYS_INLINE bool FixedBitVector::test(size_t bitIndex)
{
    if (UNLIKELY(bitIndex >= size()))
        return false;

    WordType mask = one << (bitIndex % wordSize);
    size_t wordIndex = bitIndex / wordSize;
    return m_bitVector.bits()[wordIndex] & mask;
}

ALWAYS_INLINE size_t FixedBitVector::findBit(size_t startIndex, bool value) const
{
    return m_bitVector.findBit(startIndex, value);
}

ALWAYS_INLINE unsigned FixedBitVector::hash() const
{
    return m_bitVector.hash();
}

ALWAYS_INLINE void FixedBitVector::dump(PrintStream& out) const
{
    m_bitVector.dump(out);
}

ALWAYS_INLINE void FixedBitVector::merge(const FixedBitVector& other)
{
    ASSERT(size() == other.size());
    m_bitVector.merge(other.m_bitVector);
}

ALWAYS_INLINE void FixedBitVector::filter(const FixedBitVector& other)
{
    ASSERT(size() == other.size());
    m_bitVector.filter(other.m_bitVector);
}

ALWAYS_INLINE void FixedBitVector::exclude(const FixedBitVector& other)
{
    ASSERT(size() == other.size());
    m_bitVector.exclude(other.m_bitVector);
}

struct FixedBitVectorHash {
    static unsigned hash(const FixedBitVector& vector) { return vector.hash(); }
    static bool equal(const FixedBitVector& a, const FixedBitVector& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<FixedBitVector> : FixedBitVectorHash {
};

template<> struct HashTraits<FixedBitVector> : public CustomHashTraits<FixedBitVector> {
};

} // namespace WTF

using WTF::FixedBitVector;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
