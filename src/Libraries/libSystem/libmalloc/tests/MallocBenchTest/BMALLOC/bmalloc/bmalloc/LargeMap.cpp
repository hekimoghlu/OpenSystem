/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#include "LargeMap.h"
#include <utility>

namespace bmalloc {

LargeRange LargeMap::remove(size_t alignment, size_t size)
{
    size_t alignmentMask = alignment - 1;

    LargeRange* candidate = m_free.end();
    for (LargeRange* it = m_free.begin(); it != m_free.end(); ++it) {
        if (!it->isEligibile())
            continue;

        if (it->size() < size)
            continue;

        if (candidate != m_free.end() && candidate->begin() < it->begin())
            continue;

        if (test(it->begin(), alignmentMask)) {
            char* aligned = roundUpToMultipleOf(alignment, it->begin());
            if (aligned < it->begin()) // Check for overflow.
                continue;

            char* alignedEnd = aligned + size;
            if (alignedEnd < aligned) // Check for overflow.
                continue;

            if (alignedEnd > it->end())
                continue;
        }

        candidate = it;
    }
    
    if (candidate == m_free.end())
        return LargeRange();

    return m_free.pop(candidate);
}

void LargeMap::add(const LargeRange& range)
{
    LargeRange merged = range;

    for (size_t i = 0; i < m_free.size(); ++i) {
        if (!canMerge(merged, m_free[i]))
            continue;

        merged = merge(merged, m_free.pop(i--));
    }
    
    m_free.push(merged);
}

void LargeMap::markAllAsEligibile()
{
    for (LargeRange& range : m_free)
        range.setEligible(true);
}

} // namespace bmalloc
