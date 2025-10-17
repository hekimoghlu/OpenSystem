/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

#include "Algorithm.h"
#include "Mutex.h"
#include "ObjectType.h"
#include "Sizes.h"

#if BOS(DARWIN)
#include <mach/vm_param.h>
#endif

#if !BUSE(LIBPAS)

namespace bmalloc {

class Chunk;

// Querying ObjectType for Chunk without locking.
class ObjectTypeTable {
public:
    ObjectTypeTable();

    static constexpr unsigned shiftAmount = 20;
    static_assert((1ULL << shiftAmount) == chunkSize);
    static_assert((BOS_EFFECTIVE_ADDRESS_WIDTH - shiftAmount) <= 32);

    class Bits;

    ObjectType get(Chunk*);
    void set(UniqueLockHolder&, Chunk*, ObjectType);

private:
    static unsigned convertToIndex(Chunk* chunk)
    {
        uintptr_t address = reinterpret_cast<uintptr_t>(chunk);
        BASSERT(!(address & (~chunkMask)));
        return static_cast<unsigned>(address >> shiftAmount);
    }

    Bits* m_bits;
};

class ObjectTypeTable::Bits {
public:
    using WordType = unsigned;
    static constexpr unsigned bitCountPerWord = sizeof(WordType) * 8;
    static constexpr WordType one = 1;
    constexpr Bits(Bits* previous, unsigned begin, unsigned end)
        : m_previous(previous)
        , m_begin(begin)
        , m_end(end)
    {
    }

    bool get(unsigned index);
    void set(unsigned index, bool);

    Bits* previous() const { return m_previous; }
    unsigned begin() const { return m_begin; }
    unsigned end() const { return m_end; }
    unsigned count() const { return m_end - m_begin; }
    unsigned sizeInBytes() const { return count() / 8; }

    const WordType* words() const { return const_cast<Bits*>(this)->words(); }
    WordType* words() { return reinterpret_cast<WordType*>(reinterpret_cast<uintptr_t>(this) + sizeof(Bits)); }

    WordType* wordForIndex(unsigned);

private:
    Bits* m_previous { nullptr }; // Keeping the previous Bits* just to suppress Leaks warnings.
    unsigned m_begin { 0 };
    unsigned m_end { 0 };
};
static_assert(!(sizeof(ObjectTypeTable::Bits) % sizeof(ObjectTypeTable::Bits::WordType)));

extern BEXPORT ObjectTypeTable::Bits sentinelBits;

inline ObjectTypeTable::ObjectTypeTable()
    : m_bits(&sentinelBits)
{
}

inline ObjectType ObjectTypeTable::get(Chunk* chunk)
{
    Bits* bits = m_bits;
    unsigned index = convertToIndex(chunk);
    BASSERT(bits);
    if (bits->begin() <= index && index < bits->end())
        return static_cast<ObjectType>(bits->get(index));
    return { };
}

inline bool ObjectTypeTable::Bits::get(unsigned index)
{
    unsigned n = index - begin();
    return words()[n / bitCountPerWord] & (one << (n % bitCountPerWord));
}

inline void ObjectTypeTable::Bits::set(unsigned index, bool value)
{
    unsigned n = index - begin();
    if (value)
        words()[n / bitCountPerWord] |= (one << (n % bitCountPerWord));
    else
        words()[n / bitCountPerWord] &= ~(one << (n % bitCountPerWord));
}

inline ObjectTypeTable::Bits::WordType* ObjectTypeTable::Bits::wordForIndex(unsigned index)
{
    unsigned n = index - begin();
    return &words()[n / bitCountPerWord];
}

} // namespace bmalloc

#endif
