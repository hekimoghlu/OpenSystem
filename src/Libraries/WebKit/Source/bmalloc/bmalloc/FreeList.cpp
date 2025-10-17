/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#include "BPlatform.h"
#include "FreeList.h"

#if !BUSE(TZONE)

#include "FreeListInlines.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

FreeList::FreeList() = default;

FreeList::~FreeList() = default;

void FreeList::clear()
{
    *this = FreeList();
}

void FreeList::initializeList(FreeCell* head, uintptr_t secret, unsigned bytes)
{
    // It's *slightly* more optimal to use a scrambled head. It saves a register on the fast path.
    m_scrambledHead = FreeCell::scramble(head, secret);
    m_secret = secret;
    m_payloadEnd = nullptr;
    m_remaining = 0;
    m_originalSize = bytes;
}

void FreeList::initializeBump(char* payloadEnd, unsigned remaining)
{
    m_scrambledHead = 0;
    m_secret = 0;
    m_payloadEnd = payloadEnd;
    m_remaining = remaining;
    m_originalSize = remaining;
}

bool FreeList::contains(void* target) const
{
    if (m_remaining) {
        const void* start = (m_payloadEnd - m_remaining);
        const void* end = m_payloadEnd;
        return (start <= target) && (target < end);
    }

    FreeCell* candidate = head();
    while (candidate) {
        if (static_cast<void*>(candidate) == target)
            return true;
        candidate = candidate->next(m_secret);
    }

    return false;
}

} // namespace JSC

#endif
#endif // !BUSE(TZONE)
