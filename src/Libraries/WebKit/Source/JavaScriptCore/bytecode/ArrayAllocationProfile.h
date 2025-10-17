/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

#include "IndexingType.h"
#include "JSArray.h"

namespace JSC {

class ArrayAllocationProfile {
public:
    ArrayAllocationProfile()
    {
        initializeIndexingMode(ArrayWithUndecided);
    }

    ArrayAllocationProfile(IndexingType recommendedIndexingMode)
    {
        initializeIndexingMode(recommendedIndexingMode);
    }

    IndexingType selectIndexingTypeConcurrently()
    {
        return current().indexingType();
    }

    IndexingType selectIndexingType()
    {
        ASSERT(!isCompilationThread());
        JSArray* lastArray = m_storage.pointer();
        if (lastArray && UNLIKELY(lastArray->indexingType() != current().indexingType()))
            updateProfile();
        return current().indexingType();
    }

    // vector length hint becomes [0, BASE_CONTIGUOUS_VECTOR_LEN_MAX].
    unsigned vectorLengthHintConcurrently()
    {
        return current().vectorLength();
    }

    unsigned vectorLengthHint()
    {
        ASSERT(!isCompilationThread());
        JSArray* lastArray = m_storage.pointer();
        unsigned largestSeenVectorLength = current().vectorLength();
        if (lastArray && (largestSeenVectorLength != BASE_CONTIGUOUS_VECTOR_LEN_MAX) && UNLIKELY(lastArray->getVectorLength() > largestSeenVectorLength))
            updateProfile();
        return current().vectorLength();
    }
    
    JSArray* updateLastAllocation(JSArray* lastArray)
    {
        ASSERT(!isCompilationThread());
        m_storage.setPointer(lastArray);
        return lastArray;
    }

    JS_EXPORT_PRIVATE void updateProfile();

    static IndexingType selectIndexingTypeFor(ArrayAllocationProfile* profile)
    {
        if (!profile)
            return ArrayWithUndecided;
        return profile->selectIndexingType();
    }
    
    static JSArray* updateLastAllocationFor(ArrayAllocationProfile* profile, JSArray* lastArray)
    {
        ASSERT(!isCompilationThread());
        if (profile)
            profile->updateLastAllocation(lastArray);
        return lastArray;
    }

    void initializeIndexingMode(IndexingType recommendedIndexingMode)
    {
        m_storage.setType(current().withIndexingType(recommendedIndexingMode));
    }

private:
    struct IndexingTypeAndVectorLength {
        static_assert(sizeof(IndexingType) <= sizeof(uint8_t));
        static_assert(BASE_CONTIGUOUS_VECTOR_LEN_MAX <= UINT8_MAX);

        // The format is: (IndexingType << 8) | VectorLength
        static constexpr uint16_t vectorLengthMask = static_cast<uint8_t>(-1);
        static constexpr uint16_t indexingTypeShift = 8;

    public:
        IndexingTypeAndVectorLength()
            : m_bits(0)
        {
        }

        IndexingTypeAndVectorLength(uint16_t bits)
            : m_bits(bits)
        {
        }

        IndexingTypeAndVectorLength(IndexingType indexingType, unsigned vectorLength)
            : m_bits((indexingType << indexingTypeShift) | vectorLength)
        {
            ASSERT(vectorLength <= BASE_CONTIGUOUS_VECTOR_LEN_MAX);
        }

        operator uint16_t() const { return m_bits; }

        IndexingType indexingType() const { return m_bits >> indexingTypeShift; }
        unsigned vectorLength() const { return m_bits & vectorLengthMask; }

        IndexingTypeAndVectorLength withIndexingType(IndexingType indexingType) WARN_UNUSED_RETURN
        {
            return IndexingTypeAndVectorLength(indexingType, vectorLength());
        }

    private:
        uint16_t m_bits;
    };
    
    IndexingTypeAndVectorLength current() const { return m_storage.type(); }

    using Storage = CompactPointerTuple<JSArray*, uint16_t>;
    Storage m_storage;
};

} // namespace JSC
