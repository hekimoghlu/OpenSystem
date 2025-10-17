/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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

namespace bmalloc {

template<typename Config, typename Func>
void* FreeList::allocate(const Func& slowPath)
{
    unsigned remaining = m_remaining;
    if (remaining) {
        remaining -= Config::objectSize;
        m_remaining = remaining;
        return m_payloadEnd - remaining - Config::objectSize;
    }
    
    FreeCell* result = head();
    if (!result)
        return slowPath();
    
    m_scrambledHead = result->scrambledNext;
    return result;
}

template<typename Config, typename Func>
void FreeList::forEach(const Func& func) const
{
    if (m_remaining) {
        for (unsigned remaining = m_remaining; remaining; remaining -= Config::objectSize)
            func(static_cast<void*>(m_payloadEnd - remaining));
    } else {
        for (FreeCell* cell = head(); cell;) {
            // We can use this to overwrite free objects before destroying the free list. So, we need
            // to get next before proceeding further.
            FreeCell* next = cell->next(m_secret);
            func(static_cast<void*>(cell));
            cell = next;
        }
    }
}

} // namespace bmalloc

