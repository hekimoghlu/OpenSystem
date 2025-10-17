/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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

#include "BInline.h"
#include "EligibilityResult.h"
#include "IsoAllocator.h"
#include "IsoHeapImplInlines.h"
#include "IsoPage.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename Config>
IsoAllocator<Config>::IsoAllocator(IsoHeapImpl<Config>&)
{
}

template<typename Config>
IsoAllocator<Config>::~IsoAllocator()
{
}

template<typename Config>
void* IsoAllocator<Config>::allocate(IsoHeapImpl<Config>& heap, bool abortOnFailure)
{
    static constexpr bool verbose = false;
    void* result = m_freeList.allocate<Config>(
        [&] () -> void* {
            return allocateSlow(heap, abortOnFailure);
        });
    if (verbose)
        fprintf(stderr, "%p: allocated %p of size %u\n", &heap, result, Config::objectSize);
    return result;
}

template<typename Config>
BNO_INLINE void* IsoAllocator<Config>::allocateSlow(IsoHeapImpl<Config>& heap, bool abortOnFailure)
{
    LockHolder locker(heap.lock);

    AllocationMode allocationMode = heap.updateAllocationMode();
    if (allocationMode == AllocationMode::Shared) {
        if (m_currentPage) {
            m_currentPage->stopAllocating(locker, m_freeList);
            m_currentPage = nullptr;
            m_freeList.clear();
        }
        return heap.allocateFromShared(locker, abortOnFailure);
    }

    BASSERT(allocationMode == AllocationMode::Fast);
    
    EligibilityResult<Config> result = heap.takeFirstEligible(locker);
    if (result.kind != EligibilityKind::Success) {
        RELEASE_BASSERT(result.kind == EligibilityKind::OutOfMemory);
        RELEASE_BASSERT(!abortOnFailure);
        return nullptr;
    }
    
    if (m_currentPage)
        m_currentPage->stopAllocating(locker, m_freeList);
    
    m_currentPage = result.page;
    m_freeList = m_currentPage->startAllocating(locker);
    
    return m_freeList.allocate<Config>([] () { BCRASH(); return nullptr; });
}

template<typename Config>
void IsoAllocator<Config>::scavenge(IsoHeapImpl<Config>& heap)
{
    if (m_currentPage) {
        LockHolder locker(heap.lock);
        m_currentPage->stopAllocating(locker, m_freeList);
        m_currentPage = nullptr;
        m_freeList.clear();
    }
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
