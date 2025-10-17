/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

#if !BUSE(TZONE)

#include "IsoSharedHeap.h"

#include "IsoSharedPage.h"
#include <bit>

#if !BUSE(LIBPAS)

namespace bmalloc {

template<unsigned objectSize, typename Func>
void* VariadicBumpAllocator::allocate(const Func& slowPath)
{
    unsigned remaining = m_remaining;
    if (!__builtin_usub_overflow(remaining, objectSize, &remaining)) {
        m_remaining = remaining;
        return m_payloadEnd - remaining - objectSize;
    }
    return slowPath();
}

inline constexpr unsigned computeObjectSizeForSharedCell(unsigned objectSize)
{
    return roundUpToMultipleOf<alignmentForIsoSharedAllocation>(static_cast<uintptr_t>(objectSize));
}

template<unsigned passedObjectSize>
void* IsoSharedHeap::allocateNew(bool abortOnFailure)
{
    LockHolder locker(mutex());
    constexpr unsigned objectSize = computeObjectSizeForSharedCell(passedObjectSize);
    return m_allocator.template allocate<objectSize>(
        [&] () -> void* {
            return allocateSlow<passedObjectSize>(locker, abortOnFailure);
        });
}

template<unsigned passedObjectSize>
BNO_INLINE void* IsoSharedHeap::allocateSlow(const LockHolder& locker, bool abortOnFailure)
{
    Scavenger& scavenger = *Scavenger::get();
    scavenger.scheduleIfUnderMemoryPressure(IsoSharedPage::pageSize);

    IsoSharedPage* page = IsoSharedPage::tryCreate();
    if (!page) {
        RELEASE_BASSERT(!abortOnFailure);
        return nullptr;
    }

    if (m_currentPage)
        m_currentPage->stopAllocating(locker);

    m_currentPage = page;
    m_allocator = m_currentPage->startAllocating(locker);

    constexpr unsigned objectSize = computeObjectSizeForSharedCell(passedObjectSize);
    return m_allocator.allocate<objectSize>([] () { BCRASH(); return nullptr; });
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
