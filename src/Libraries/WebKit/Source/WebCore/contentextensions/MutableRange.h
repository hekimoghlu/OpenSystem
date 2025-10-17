/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore {

namespace ContentExtensions {

template <typename CharacterType, typename DataType>
class MutableRange {
    typedef MutableRange<CharacterType, DataType> TypedMutableRange;
public:
    MutableRange(uint32_t nextRangeIndex, CharacterType first, CharacterType last)
        : nextRangeIndex(nextRangeIndex)
        , first(first)
        , last(last)
    {
        ASSERT(first <= last);
    }

    MutableRange(const DataType& data, uint32_t nextRangeIndex, CharacterType first, CharacterType last)
        : data(data)
        , nextRangeIndex(nextRangeIndex)
        , first(first)
        , last(last)
    {
        ASSERT(first <= last);
    }

    MutableRange(DataType&& data, uint32_t nextRangeIndex, CharacterType first, CharacterType last)
        : data(WTFMove(data))
        , nextRangeIndex(nextRangeIndex)
        , first(first)
        , last(last)
    {
        ASSERT(first <= last);
    }

    MutableRange(MutableRange&& other)
        : data(WTFMove(other.data))
        , nextRangeIndex(other.nextRangeIndex)
        , first(other.first)
        , last(other.last)
    {
        ASSERT(first <= last);
    }

    TypedMutableRange& operator=(TypedMutableRange&& other)
    {
        data = WTFMove(other.data);
        nextRangeIndex = WTFMove(other.nextRangeIndex);
        first = WTFMove(other.first);
        last = WTFMove(other.last);
        return *this;
    }

    DataType data;

    // We use a funny convention: if there are no nextRange, the nextRangeIndex is zero.
    // This is faster to check than a special value in many cases.
    // We can use zero because ranges can only form a chain, and the first range is always zero by convention.
    // When we insert something in from of the first range, we swap the values.
    uint32_t nextRangeIndex;
    CharacterType first;
    CharacterType last;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
