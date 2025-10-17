/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include "ObjectTypeTable.h"

#include "VMAllocate.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

ObjectTypeTable::Bits sentinelBits { nullptr, 0, 0 };

void ObjectTypeTable::set(UniqueLockHolder&, Chunk* chunk, ObjectType objectType)
{
    unsigned index = convertToIndex(chunk);
    Bits* bits = m_bits;
    if (!(bits->begin() <= index && index < bits->end())) {
        unsigned newBegin = 0;
        unsigned newEnd = 0;
        if (bits == &sentinelBits) {
            // This is initial allocation of ObjectTypeTable. In this case, it could be possible that for the first registration,
            // some VAs are already allocated for a different purpose, and later they will be reused for bmalloc. In that case,
            // soon, we will see a smaller index request than this initial one. We try to subtract a 128MB offset to the initial
            // newBegin to cover such patterns without extending table too quickly, and if we can't subtract 128MB, we will set
            // newBegin to 0.  
            constexpr unsigned offsetForInitialAllocation = ObjectTypeTable::Bits::bitCountPerWord * 4;
            if (index < offsetForInitialAllocation)
                newBegin = 0;
            else
                newBegin = index - offsetForInitialAllocation;
            newEnd = index + 1;
        } else if (index < bits->begin()) {
            BASSERT(bits->begin());
            BASSERT(bits->end());
            // We need to verify if "bits->begin() - bits->count()" doesn't underflow,
            // otherwise we will set "newBegin" as "index" and it creates a pathological
            // case that will keep increasing BitVector everytime we access
            // "index < bits->begin()".
            if (bits->begin() < bits->count())
                newBegin = 0;
            else
                newBegin = std::min<unsigned>(index, bits->begin() - bits->count());
            newEnd = bits->end();
        } else {
            BASSERT(bits->begin());
            BASSERT(bits->end());
            newBegin = bits->begin();
            // We need to verify if "bits->end() + bits->count()" doesn't overflow,
            // otherwise we will set "newEnd" as "index + 1" and it creates a
            // pathological case that will keep increasing BitVector everytime we access
            // "index > bits->end()".
            if (std::numeric_limits<unsigned>::max() - bits->count() < bits->end())
                newEnd = std::numeric_limits<unsigned>::max();
            else
                newEnd = std::max<unsigned>(index + 1, bits->end() + bits->count());
        }
        newBegin = static_cast<unsigned>(roundDownToMultipleOf<size_t>(ObjectTypeTable::Bits::bitCountPerWord, newBegin));
        BASSERT(newEnd > newBegin);

        unsigned count = newEnd - newBegin;
        size_t size = vmSize(sizeof(Bits) + (roundUpToMultipleOf<size_t>(ObjectTypeTable::Bits::bitCountPerWord, count) / 8));
        RELEASE_BASSERT(size <= 0x80000000U); // Too large bitvector, out-of-memory.
        size = roundUpToPowerOfTwo(size);
        newEnd = newBegin + ((size - sizeof(Bits)) / sizeof(ObjectTypeTable::Bits::WordType)) * ObjectTypeTable::Bits::bitCountPerWord;
        BASSERT(newEnd > newBegin);
        void* allocated = vmAllocate(size);
        memset(allocated, 0, size);
        auto* newBits = new (allocated) Bits(bits, newBegin, newEnd);

        memcpy(newBits->wordForIndex(bits->begin()), bits->words(), bits->sizeInBytes());
#if !defined(NDEBUG)
        for (unsigned index = bits->begin(); index < bits->end(); ++index)
            BASSERT(bits->get(index) == newBits->get(index));
#endif
        std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure table gets valid when it is visible to the other threads since ObjectTypeTable::get does not take a lock.
        m_bits = newBits;
        bits = newBits;
    }
    bool value = !!static_cast<std::underlying_type_t<ObjectType>>(objectType);
    BASSERT(static_cast<ObjectType>(value) == objectType);
    bits->set(index, value);
}

} // namespace bmalloc

#endif
