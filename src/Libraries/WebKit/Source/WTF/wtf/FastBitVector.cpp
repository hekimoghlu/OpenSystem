/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include "config.h"
#include <wtf/FastBitVector.h>

#include <wtf/NeverDestroyed.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(FastBitVector);

void FastBitVectorWordOwner::setEqualsSlow(const FastBitVectorWordOwner& other)
{
    if (m_words)
        FastBitVectorMalloc::free(m_words);
    m_words = static_cast<uint32_t*>(FastBitVectorMalloc::malloc(other.arrayLength() * sizeof(uint32_t)));
    m_numBits = other.m_numBits;
    memcpySpan(wordsSpan(), other.wordsSpan());
}

void FastBitVectorWordOwner::resizeSlow(size_t numBits)
{
    size_t newLength = fastBitVectorArrayLength(numBits);
    size_t oldLength = arrayLength();
    RELEASE_ASSERT(newLength >= oldLength);
    
    // Use fastMalloc instead of fastRealloc because we expect the common
    // use case for this method to be initializing the size of the bitvector.
    
    auto newArray = unsafeMakeSpan(static_cast<uint32_t*>(FastBitVectorMalloc::malloc(newLength * sizeof(uint32_t))), newLength);
    memcpySpan(newArray, wordsSpan());
    zeroSpan(newArray.subspan(oldLength));
    if (m_words)
        FastBitVectorMalloc::free(m_words);
    m_words = newArray.data();
}

void FastBitVector::clearRange(size_t begin, size_t end)
{
    if (end - begin < 32) {
        for (size_t i = begin; i < end; ++i)
            at(i) = false;
        return;
    }
    
    size_t endBeginSlop = (begin + 31) & ~31;
    size_t beginEndSlop = end & ~31;
    
    for (size_t i = begin; i < endBeginSlop; ++i)
        at(i) = false;
    for (size_t i = beginEndSlop; i < end; ++i)
        at(i) = false;
    for (size_t i = endBeginSlop / 32; i < beginEndSlop / 32; ++i)
        m_words.word(i) = 0;
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
