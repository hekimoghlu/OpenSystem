/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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

#include "FreeList.h"
#include "MarkedBlock.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template<typename Func>
ALWAYS_INLINE HeapCell* FreeList::allocateWithCellSize(const Func& slowPath, size_t cellSize)
{
    if (LIKELY(m_intervalStart < m_intervalEnd)) {
        char* result = m_intervalStart;
        m_intervalStart += cellSize;
        return std::bit_cast<HeapCell*>(result);
    }
    
    FreeCell* cell = nextInterval();
    if (UNLIKELY(isSentinel(cell)))
        return slowPath();

    FreeCell::advance(m_secret, m_nextInterval, m_intervalStart, m_intervalEnd);
    
    // It's an invariant of our allocator that we don't create empty intervals, so there 
    // should always be enough space remaining to allocate a cell.
    char* result = m_intervalStart;
    m_intervalStart += cellSize;
    return std::bit_cast<HeapCell*>(result);
}

template<typename Func>
void FreeList::forEach(const Func& func) const
{
    FreeCell* cell = nextInterval();
    char* intervalStart = m_intervalStart;
    char* intervalEnd = m_intervalEnd;
    ASSERT(intervalEnd - intervalStart < (ptrdiff_t)(16 * KB));

    while (true) {
        for (; intervalStart < intervalEnd; intervalStart += m_cellSize)
            func(std::bit_cast<HeapCell*>(intervalStart));

        // If we explore the whole interval and the cell is the sentinel value, though, we should
        // immediately exit so we don't decode anything out of bounds.
        if (isSentinel(cell))
            break;

        FreeCell::advance(m_secret, cell, intervalStart, intervalEnd);
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
