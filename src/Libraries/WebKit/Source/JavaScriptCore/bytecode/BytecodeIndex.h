/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include <wtf/HashTraits.h>
#include <wtf/MathExtras.h>

namespace WTF {
class PrintStream;
}

namespace JSC {

using Checkpoint = uint8_t;
static constexpr Checkpoint noCheckpoints = 0;

class BytecodeIndex {
public:
    BytecodeIndex() = default;
    BytecodeIndex(WTF::HashTableDeletedValueType)
        : m_packedBits(deletedValue().asBits())
    {
    }

    explicit BytecodeIndex(uint32_t bytecodeOffset, Checkpoint checkpoint = noCheckpoints)
        : m_packedBits(pack(bytecodeOffset, checkpoint))
    {
        ASSERT(*this);
    }

    static constexpr uint32_t numberOfCheckpoints = 4;
    static_assert(hasOneBitSet(numberOfCheckpoints), "numberOfCheckpoints should be a power of 2");
    static constexpr uint32_t checkpointMask = numberOfCheckpoints - 1;
    static constexpr uint32_t checkpointShift = WTF::getMSBSetConstexpr(numberOfCheckpoints);

    uint32_t offset() const { return m_packedBits >> checkpointShift; }
    Checkpoint checkpoint() const { return m_packedBits & checkpointMask; }
    uint32_t asBits() const { return m_packedBits; }

    unsigned hash() const { return intHash(m_packedBits); }
    static BytecodeIndex deletedValue() { return fromBits(invalidOffset - 1); }
    bool isHashTableDeletedValue() const { return *this == deletedValue(); }

    static BytecodeIndex fromBits(uint32_t bits);
    BytecodeIndex withCheckpoint(Checkpoint checkpoint) const { return BytecodeIndex(offset(), checkpoint); }

    // Comparison operators.
    explicit operator bool() const { return m_packedBits != invalidOffset && m_packedBits != deletedValue().offset(); }
    bool operator ==(const BytecodeIndex& other) const { return asBits() == other.asBits(); }

    bool operator <(const BytecodeIndex& other) const { return asBits() < other.asBits(); }
    bool operator >(const BytecodeIndex& other) const { return asBits() > other.asBits(); }
    bool operator <=(const BytecodeIndex& other) const { return asBits() <= other.asBits(); }
    bool operator >=(const BytecodeIndex& other) const { return asBits() >= other.asBits(); }


    void dump(WTF::PrintStream&) const;

private:
    static constexpr uint32_t invalidOffset = std::numeric_limits<uint32_t>::max();

    static uint32_t pack(uint32_t bytecodeIndex, Checkpoint);

    uint32_t m_packedBits { invalidOffset };
};

inline uint32_t BytecodeIndex::pack(uint32_t bytecodeIndex, Checkpoint checkpoint)
{
    ASSERT(checkpoint < numberOfCheckpoints);
    ASSERT((bytecodeIndex << checkpointShift) >> checkpointShift == bytecodeIndex);
    return bytecodeIndex << checkpointShift | checkpoint;
}

inline BytecodeIndex BytecodeIndex::fromBits(uint32_t bits)
{
    BytecodeIndex result;
    result.m_packedBits = bits;
    return result;
}

struct BytecodeIndexHash {
    static unsigned hash(const BytecodeIndex& key) { return key.hash(); }
    static bool equal(const BytecodeIndex& a, const BytecodeIndex& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::BytecodeIndex> : JSC::BytecodeIndexHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::BytecodeIndex> : SimpleClassHashTraits<JSC::BytecodeIndex> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF
